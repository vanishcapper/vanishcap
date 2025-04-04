"""Unit tests for logging utilities."""

import logging
import unittest
from io import StringIO
from unittest.mock import patch

from vanishcap.utils.logging import WorkerNameFilter, get_worker_logger


class TestLoggingUtils(unittest.TestCase):
    """Test cases for logging utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a string buffer to capture log output
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.addFilter(WorkerNameFilter())
        self.handler.setFormatter(logging.Formatter("%(level_short)s - %(worker_name)s %(message)s"))

    def test_worker_name_filter(self):
        """Test the WorkerNameFilter class."""
        # Create a log record
        record = logging.LogRecord(
            "vanishcap.workers.test_worker",  # name
            logging.WARNING,  # level
            "test.py",  # pathname
            10,  # lineno
            "Test message",  # msg
            (),  # args
            None,  # exc_info
        )

        # Apply filter
        name_filter = WorkerNameFilter()
        name_filter.filter(record)

        # Check worker name was extracted and padded
        self.assertEqual(record.worker_name, "test_worker")
        self.assertEqual(len(record.worker_name), 10)
        self.assertEqual(record.level_short, "WW")

    def test_get_worker_logger_default_level(self):
        """Test getting a worker logger with default log level."""
        logger = get_worker_logger("test_worker")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(logger.name, "vanishcap.workers.test_worker")
        self.assertTrue(logger.handlers)  # Should have at least one handler

    def test_get_worker_logger_custom_level(self):
        """Test getting a worker logger with custom log level."""
        logger = get_worker_logger("test_worker", "DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_worker_logger_invalid_level(self):
        """Test getting a worker logger with invalid log level."""
        logger = get_worker_logger("test_worker", "INVALID")
        self.assertEqual(logger.level, logging.INFO)  # Should default to INFO

    def test_logger_output_format(self):
        """Test the format of logger output."""
        logger = logging.getLogger("test")
        logger.addHandler(self.handler)
        logger.setLevel(logging.INFO)

        # Log a test message
        logger.warning("Test message")

        # Check output format
        output = self.log_output.getvalue().strip()
        self.assertIn("WW", output)  # Level short
        self.assertIn("test      ", output)  # Padded worker name
        self.assertIn("Test message", output)  # Message

    def test_multiple_logger_instances(self):
        """Test that multiple loggers for the same worker share handlers."""
        logger1 = get_worker_logger("test_worker")
        logger2 = get_worker_logger("test_worker")

        # Should have the same handlers
        self.assertEqual(logger1.handlers, logger2.handlers)

    def test_different_log_levels(self):
        """Test logger behavior with different log levels."""
        logger = logging.getLogger("test")
        logger.addHandler(self.handler)
        logger.setLevel(logging.WARNING)

        # Debug shouldn't be logged
        logger.debug("Debug message")
        self.assertEqual(self.log_output.getvalue(), "")

        # Warning should be logged
        logger.warning("Warning message")
        self.assertIn("Warning message", self.log_output.getvalue())

    def test_worker_name_padding(self):
        """Test worker name padding for different length names."""
        # Test short name
        record = logging.LogRecord(
            "vanishcap.workers.a", logging.INFO, "test.py", 10, "Test message", (), None  # Short name
        )
        name_filter = WorkerNameFilter()
        name_filter.filter(record)
        self.assertEqual(len(record.worker_name), 10)

        # Test long name
        record = logging.LogRecord(
            "vanishcap.workers.very_long_name", logging.INFO, "test.py", 10, "Test message", (), None  # Long name
        )
        name_filter.filter(record)
        self.assertEqual(len(record.worker_name), 10)


if __name__ == "__main__":
    unittest.main()
