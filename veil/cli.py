"""Command line interface for veil."""

import click
import signal
import traceback
import time
from pathlib import Path

from veil.controller import Controller
from veil.event import Event
from veil.utils.logging import get_worker_logger


@click.group(invoke_without_command=True)
@click.argument("config", type=click.Path(exists=True))
def cli(config: str):
    """Run the veil pipeline with the specified config file.

    Args:
        config: Path to the YAML configuration file
    """
    logger = get_worker_logger("cli", "INFO")

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping...")
        if controller is not None:
            controller.stop()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    controller = None
    try:
        # Create and start the controller
        controller = Controller(config)
        controller.start()

        # Wait for signal or all workers to stop
        while True:
            # Check if all workers have stopped
            all_stopped = True
            for worker in controller.workers.values():
                if worker._run_thread is not None and worker._run_thread.is_alive():
                    all_stopped = False
                    break

            if all_stopped:
                logger.info("All workers have stopped, exiting...")
                break

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)  # 1ms sleep

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Main loop stopped")
        if controller is not None:
            controller.stop()