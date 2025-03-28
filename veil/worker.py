"""Base class for all workers in the system."""

import queue
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from veil.event import Event
from veil.utils.logging import get_worker_logger


class Worker(ABC):
    """Base class for all workers in the system."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize the worker.

        Args:
            name: Name of the worker
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = get_worker_logger(name, config.get("log_level"))
        self.logger.info(f"Initialized {name} worker")

        # Thread control
        self._run_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Event handling
        self._event_queue = queue.Queue()  # Queue for receiving events
        self._controller: Optional[Any] = None  # Reference to the Controller

        # Profiling
        self._last_task_time = 0.0  # Time taken by last task execution
        self._max_task_time = 0.0  # Maximum task time in the last window
        self._profile_window = config.get("profile_window", 1.0)  # Window size in seconds for max time
        self._window_start = time.time()  # Start of the current window
        self.logger.info(f"Using profile window of {self._profile_window:.1f}s")

    def _get_max_task_time(self) -> float:
        """Get the maximum task time over the last window.

        Returns:
            float: Maximum task time in seconds
        """
        current_time = time.time()

        # If we've moved past the window, reset the max
        if current_time - self._window_start > self._profile_window:
            self._max_task_time = 0.0
            self._window_start = current_time

        return self._max_task_time

    def start(self, controller: Any, run_in_main_thread: bool = False) -> None:
        """Start the worker.

        Args:
            controller: Reference to the Controller for event routing
            run_in_main_thread: If True, run the worker in the main thread
        """
        self.logger.info(f"Starting {self.name} worker")
        self._stop_event.clear()
        self._controller = controller

        if not run_in_main_thread:
            self._run_thread = threading.Thread(target=self._run, daemon=True)
            self._run_thread.start()
        else:
            # For main thread workers, process events in the main loop
            self._run_with_events()

    def _run_with_events(self) -> None:
        """Run the worker in the main thread, processing events."""
        while not self._stop_event.is_set():
            # Process any pending events
            try:
                event = self._event_queue.get_nowait()
                self.logger.debug(f"Worker {self.name} processing event {event.event_name}")
                self(event)
                # If stop event was received, break immediately
                if self._stop_event.is_set():
                    break
            except queue.Empty:
                pass

            # Run one iteration of the main loop with timing
            start_time = time.perf_counter()
            self._task()
            self._last_task_time = time.perf_counter() - start_time

            # Update max time if this task took longer
            self._max_task_time = max(self._max_task_time, self._last_task_time)

            # Get and log max time
            max_time = self._get_max_task_time()
            self.logger.debug(f"Worker {self.name} max task execution time: {max_time:.3f}s")

            # Emit profiling event with max time
            self._emit(Event(
                worker_name=self.name,
                event_name="worker_profile",
                data={"task_time": max_time}
            ))

            # If stop event was received, break immediately
            if self._stop_event.is_set():
                break

        # Process any remaining events before finishing
        try:
            while True:
                event = self._event_queue.get_nowait()
                self.logger.debug(f"Worker {self.name} processing remaining event {event.event_name}")
                self(event)
        except queue.Empty:
            pass

        # Clean up resources when stopping
        self._finish()
        self.logger.info(f"{self.name} loop stopped")

    @abstractmethod
    def _task(self) -> None:
        """Run one iteration of the worker's task.

        This method should be implemented by each worker to perform its specific task.
        """
        pass

    def _finish(self) -> None:
        """Clean up resources when the worker is stopping.

        This method should be implemented by each worker to clean up its resources.
        """
        pass

    def _run(self) -> None:
        """Main run loop for the worker.

        This method is common to all workers and handles the main event loop.
        """
        self.logger.info(f"Starting {self.name} loop")

        try:
            while not self._stop_event.is_set():
                # Process any pending events
                try:
                    event = self._event_queue.get_nowait()
                    self.logger.debug(f"Worker {self.name} processing event {event.event_name}")
                    self(event)
                except queue.Empty:
                    pass

                # Run one iteration of the main loop with timing
                start_time = time.perf_counter()
                self._task()
                self._last_task_time = time.perf_counter() - start_time

                # Update max time if this task took longer
                self._max_task_time = max(self._max_task_time, self._last_task_time)

                # Get and log max time
                max_time = self._get_max_task_time()
                self.logger.debug(f"Worker {self.name} max task execution time: {max_time:.3f}s")

                # Emit profiling event with max time
                self._emit(Event(
                    worker_name=self.name,
                    event_name="worker_profile",
                    data={"task_time": max_time}
                ))

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # Reduced sleep time to process frames more frequently
        except Exception as e:
            self.logger.error(f"Error in {self.name} loop: {e}")
            traceback.print_exc()
        finally:
            # Process any remaining events before finishing
            try:
                while True:
                    event = self._event_queue.get_nowait()
                    self.logger.debug(f"Worker {self.name} processing remaining event {event.event_name}")
                    self(event)
            except queue.Empty:
                pass

            self._finish()
            self.logger.info(f"{self.name} loop stopped")

    def stop(self) -> None:
        """Stop the worker."""
        self.logger.debug(f"Worker {self.name} stopping")
        self.logger.info(f"Stopping {self.name} worker")
        self._stop_event.set()

        # Wait for thread to finish
        if self._run_thread is not None and self._run_thread.is_alive():
            self.logger.debug(f"Worker {self.name} joining run thread")
            self._run_thread.join()

        # Clean up resources
        self.logger.debug(f"Worker {self.name} calling _finish")
        self._finish()
        self.logger.debug(f"Worker {self.name} stopped")

    def _dispatch(self, event: Event) -> None:
        """Handle an incoming event from the Controller.

        This is the entry point for all events coming from the Controller.
        By default, it just puts the event in the queue, but subclasses can override
        this to add special handling.

        Args:
            event: Event to handle
        """
        self.logger.debug(f"Worker {self.name} dispatching event {event.event_name}")
        self._event_queue.put(event)
        self.logger.debug(f"Worker {self.name} event {event.event_name} queued")

    def _emit(self, event: Event) -> None:
        """Emit an event to the Controller.

        Args:
            event: Event to emit
        """
        if self._controller is not None:
            self.logger.debug(f"Worker {self.name} emitting event {event.event_name}")
            self._controller(event)
            self.logger.debug(f"Worker {self.name} event {event.event_name} routed")

    @abstractmethod
    def __call__(self, event: Event) -> None:
        """Handle an event. Must be implemented by subclasses.

        Args:
            event: Event to handle
        """
        pass