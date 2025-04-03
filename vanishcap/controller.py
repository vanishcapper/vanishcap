"""Controller for managing worker threads and event routing."""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Set

from omegaconf import OmegaConf
from vanishcap.event import Event
from vanishcap.worker import Worker
from vanishcap.utils.logging import get_worker_logger
from vanishcap.utils.wifi import WifiManager, WifiError


class InitializationError(Exception):
    """Exception raised when controller initialization fails."""


class Controller:
    """Controller for managing worker threads and event routing."""

    def __init__(self, config_path: str) -> None:
        """Initialize the controller.

        Args:
            config_path: Path to the configuration file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid
            WifiError: If WiFi initialization fails
            InitializationError: If any other initialization fails
        """
        # Load config first to get controller's log level
        self.config = self._load_config(config_path)

        # Initialize logger with config's log level
        controller_config = self.config.get("controller", {})
        log_level = controller_config.get("log_level", "WARNING")
        self.logger = get_worker_logger("controller", log_level)
        self.logger.warning("Initializing controller with config: %s", config_path)

        # Initialize WiFi manager if not in offline mode
        self.wifi_manager = None
        if not controller_config.get("offline", False):
            try:
                wifi_config = controller_config.get("wifi", {})
                wifi_config["log_level"] = log_level  # Pass controller's log level
                self.wifi_manager = WifiManager(wifi_config)
            except WifiError as e:
                self.logger.error("WiFi initialization failed: %s", e)
                raise InitializationError(f"WiFi initialization failed: {e}") from e
        else:
            self.logger.warning("Running in offline mode - skipping WiFi management")

        self.workers: Dict[str, Worker] = {}
        self.event_routes: Dict[str, Set[str]] = {}

        # Initialize workers in dependency order
        try:
            self._init_workers()
        except Exception as e:
            self.logger.error("Failed to initialize workers: %s", e)
            raise InitializationError(f"Failed to initialize workers: {e}") from e

        # Build event routing map
        self._build_event_routes()

        self.logger.warning("Controller initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            config = OmegaConf.load(config_path)
            return OmegaConf.to_container(config)
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}") from e

    def _can_init_worker(self, worker_name: str, initialized_workers: Set[str]) -> bool:
        """Check if a worker's dependencies have been initialized.

        Args:
            worker_name: Name of the worker to check
            initialized_workers: Set of already initialized workers

        Returns:
            bool: True if all dependencies are initialized, False otherwise
        """
        worker_config = self.config[worker_name]
        dependencies = worker_config.get("depends_on", [])

        # Check if all dependencies have been initialized
        return all(dep in initialized_workers for dep in dependencies)

    def _init_workers(self) -> None:
        """Initialize all workers in dependency order."""
        # Import the workers module
        worker_module = importlib.import_module("vanishcap.workers")

        # Get all worker classes from the module
        worker_classes = {
            name.lower(): cls
            for name, cls in inspect.getmembers(worker_module)
            if inspect.isclass(cls) and issubclass(cls, Worker) and cls != Worker
        }

        # Get list of worker sections (excluding controller)
        worker_sections = [name for name in self.config.keys() if name != "controller"]

        # Initialize workers in dependency order
        initialized_workers: Set[str] = set()
        while len(initialized_workers) < len(worker_sections):
            initialized_any = False
            for worker_type in worker_sections:
                # Skip already initialized workers
                if worker_type in initialized_workers:
                    continue

                # Check if dependencies are satisfied
                if self._can_init_worker(worker_type, initialized_workers):
                    worker_type_lower = worker_type.lower()
                    if worker_type_lower not in worker_classes:
                        available = ", ".join(sorted(worker_classes.keys()))
                        raise ValueError(f"Unknown worker type '{worker_type}'. Available workers: {available}")

                    worker_class = worker_classes[worker_type_lower]
                    # Initialize the worker
                    self.workers[worker_type] = worker_class(self.config[worker_type])
                    initialized_workers.add(worker_type)
                    initialized_any = True

            if not initialized_any:
                # If we couldn't initialize any workers, we have a circular dependency
                uninitialized = set(worker_sections) - initialized_workers
                raise ValueError(f"Circular dependency detected among workers: {uninitialized}")

    def _build_event_routes(self) -> None:
        """Build the event routing map from config.

        Each worker in the config has an "events" list containing "worker_name": "event_name" pairs
        that specify which events from which workers it wants to receive.
        """
        # Don't add stop events to routing - they're handled directly
        for worker_name, worker_config in self.config.items():
            # Skip the controller section
            if worker_name == "controller":
                continue

            worker_lower = worker_name.lower()
            if worker_lower not in self.workers:
                raise ValueError(f"Unknown worker: {worker_name}")

            # Skip routes that are just "run"
            if isinstance(worker_config, str) and worker_config == "run":
                continue

            # Get the events list for this worker
            events = worker_config.get("events", [])
            if not isinstance(events, list):
                raise ValueError(f"Invalid events list for {worker_name}: {events}")

            # Add routes for each event this worker wants to receive
            for event_spec in events:
                if not isinstance(event_spec, dict):
                    raise ValueError(f"Invalid event spec for {worker_name}: {event_spec}")

                # Each event spec should have a single key:value pair
                if len(event_spec) != 1:
                    raise ValueError(
                        f"Event spec for {worker_name} should have exactly one key:value pair: {event_spec}"
                    )

                source_worker, event_name = next(iter(event_spec.items()))
                source_worker_lower = source_worker.lower()

                # Validate source worker exists
                if source_worker_lower not in self.workers:
                    raise ValueError(f"Unknown source worker in event spec for {worker_name}: {source_worker}")

                # Add the route
                if event_name not in self.event_routes:
                    self.event_routes[event_name] = set()
                self.event_routes[event_name].add(worker_lower)

    def start(self) -> None:
        """Start all workers."""
        from vanishcap.workers.ui import Ui  # pylint: disable=import-outside-toplevel  # Import here to avoid circular imports

        # Find UI worker if it exists
        ui_worker = None
        for name, worker in self.workers.items():
            if isinstance(worker, Ui):
                ui_worker = (name, worker)
                break

        # Start non-UI workers first
        for name, worker in self.workers.items():
            if not isinstance(worker, Ui):
                self.logger.debug("Starting worker: %s", name)
                worker.start(self)

        # Start UI worker in main thread if it exists
        if ui_worker is not None:
            name, worker = ui_worker
            self.logger.debug("Starting UI worker: %s", name)
            worker.start(self, run_in_main_thread=True)

    def stop(self) -> None:
        """Stop all workers."""
        self.logger.debug("Stopping all workers")
        for name, worker in self.workers.items():
            self.logger.debug("Stopping worker: %s", name)
            worker.stop()
        self.logger.debug("All workers stopped")

    def __call__(self, event: Event) -> None:
        """Handle an event by routing it to appropriate workers.

        Args:
            event: Event to handle
        """
        self.logger.debug("Controller received event: %s from %s", event.event_name, event.worker_name)

        # Handle stop event from any worker by stopping all workers
        if event.event_name == "stop":
            self.logger.debug("Controller received stop event from %s, stopping all workers", event.worker_name)
            self.stop()
            return

        # Get target workers for this event type
        targets = self.event_routes.get(event.event_name, set())
        self.logger.debug("Routing event %s to targets: %s", event.event_name, targets)

        # Route event to each target worker
        for target in targets:
            target_lower = target.lower()
            if target_lower in self.workers:
                self.logger.debug("Controller routing event to worker: %s", target)
                self.workers[target]._dispatch(event)
            else:
                self.logger.warning("Unknown target worker: %s", target)

    def __enter__(self) -> "Controller":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

        # Restore previous WiFi connection if needed
        if self.wifi_manager and not self.config.get("controller", {}).get("offline", False):
            self.logger.info("Restoring previous WiFi connection")
            self.wifi_manager.reconnect_previous()
