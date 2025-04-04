"""Unit tests for the CLI module."""

# pylint: disable=protected-access,unused-argument,too-many-instance-attributes

import os
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import yaml
from click.testing import CliRunner

from vanishcap.cli import cli
from vanishcap.controller import InitializationError


class TestCLI(unittest.TestCase):
    """Test cases for the command line interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create a temporary config file
        self.config = {"controller": {"offline": True, "log_level": "DEBUG"}, "test_worker": {"events": []}}

        # Create temp config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temp config file
        os.unlink(self.config_path)
        os.rmdir(self.temp_dir)

    @patch("vanishcap.cli.Controller")
    def test_cli_normal_execution(self, mock_controller_class):
        """Test normal CLI execution."""
        # Mock controller instance
        mock_controller = MagicMock()
        mock_controller_class.return_value = mock_controller

        # Mock worker thread that stops after a short time
        mock_worker = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.side_effect = [True, True, False]  # Stop after 3 checks
        mock_worker._run_thread = mock_thread
        mock_controller.workers = {"test_worker": mock_worker}

        # Run CLI
        result = self.runner.invoke(cli, [self.config_path])

        # Check CLI execution
        self.assertEqual(result.exit_code, 0)
        mock_controller_class.assert_called_once_with(self.config_path)
        mock_controller.start.assert_called_once()
        mock_controller.stop.assert_called_once()

    @patch("vanishcap.cli.Controller")
    def test_cli_initialization_error(self, mock_controller_class):
        """Test CLI behavior when controller initialization fails."""
        # Make controller raise initialization error
        mock_controller_class.side_effect = InitializationError("Test error")

        # Run CLI
        result = self.runner.invoke(cli, [self.config_path])

        # Check error handling
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Test error", result.output)

    def test_cli_missing_config(self):
        """Test CLI behavior with missing config file."""
        # Run CLI with non-existent config
        result = self.runner.invoke(cli, ["nonexistent.yaml"])

        # Check error handling
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)

    @patch("vanishcap.cli.Controller")
    def test_cli_signal_handling(self, mock_controller_class):
        """Test CLI signal handling."""
        # Mock controller instance
        mock_controller = MagicMock()
        mock_controller_class.return_value = mock_controller

        # Mock worker that keeps running until signal
        mock_worker = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.side_effect = lambda: not mock_controller.stop.called
        mock_worker._run_thread = mock_thread
        mock_controller.workers = {"test_worker": mock_worker}

        # Create a threading event to coordinate test
        signal_sent = threading.Event()

        # Store original signal handler
        original_handler = signal.getsignal(signal.SIGINT)

        def signal_handler(signum, frame):
            """Handle SIGINT by stopping the controller."""
            mock_controller.stop()
            signal_sent.set()

        try:
            # Set up signal handler in main thread
            signal.signal(signal.SIGINT, signal_handler)

            # Start CLI in a thread
            cli_thread = threading.Thread(target=lambda: self.runner.invoke(cli, [self.config_path]))
            cli_thread.start()

            # Give it time to start
            time.sleep(0.1)

            # Send SIGINT
            os.kill(os.getpid(), signal.SIGINT)

            # Wait for signal handler to complete
            signal_sent.wait(timeout=1.0)

            # Wait for thread to finish
            cli_thread.join(timeout=1.0)

            # Check that controller was stopped
            mock_controller.stop.assert_called_once()
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

    @patch("vanishcap.cli.Controller")
    def test_cli_runtime_error(self, mock_controller_class):
        """Test CLI behavior when runtime error occurs."""
        # Mock controller instance that raises an error during start
        mock_controller = MagicMock()
        mock_controller.start.side_effect = RuntimeError("Test runtime error")
        mock_controller_class.return_value = mock_controller

        # Run CLI and expect RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            with self.runner.isolated_filesystem():
                with open("test_config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(self.config, f)
                self.runner.invoke(cli, ["test_config.yaml"], catch_exceptions=False)

        # Check error message
        self.assertEqual(str(cm.exception), "Test runtime error")
        mock_controller.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
