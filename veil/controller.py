"""Controller for managing worker threads and event routing."""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from omegaconf import OmegaConf
from veil.event import Event
from veil.worker import Worker
from veil.workers.ui import Ui
from veil.utils.logging import get_worker_logger


class Controller:
    """Controller for managing worker threads and event routing."""

    def __init__(self, config_path: str) -> None:
        """Initialize the controller.

        Args:
            config_path: Path to the configuration file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid
        """
        self.logger = get_worker_logger("controller", "INFO")
        self.logger.info(f"Initializing controller with config: {config_path}")

        self.config = self._load_config(config_path)
        self.workers: Dict[str, Worker] = {}
        self.event_routes: Dict[str, Set[str]] = {}

        # Initialize workers in dependency order
        self._init_workers()

        # Build event routing map
        self._build_event_routes()

        self.logger.info("Controller initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dict containing the configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            config = OmegaConf.load(config_path)
            return OmegaConf.to_container(config)
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")

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
        worker_module = importlib.import_module("veil.workers")

        # Get all worker classes from the module
        worker_classes = {
            name.lower(): cls for name, cls in inspect.getmembers(worker_module)
            if inspect.isclass(cls) and issubclass(cls, Worker) and cls != Worker
        }

        # Initialize workers in dependency order
        initialized_workers: Set[str] = set()
        while len(initialized_workers) < len(self.config):
            initialized_any = False
            for worker_type, worker_config in self.config.items():
                # Skip already initialized workers
                if worker_type in initialized_workers:
                    continue

                # Check if dependencies are satisfied
                if self._can_init_worker(worker_type, initialized_workers):
                    worker_type_lower = worker_type.lower()
                    if worker_type_lower not in worker_classes:
                        available = ", ".join(sorted(worker_classes.keys()))
                        raise ValueError(
                            f"Unknown worker type '{worker_type}'. Available workers: {available}"
                        )

                    worker_class = worker_classes[worker_type_lower]
                    # Initialize the worker
                    self.workers[worker_type] = worker_class(worker_config)
                    initialized_workers.add(worker_type)
                    initialized_any = True

            # If we couldn't initialize any workers and we're not done, we have a circular dependency
            if not initialized_any:
                remaining = [name for name in self.config.keys()
                           if name not in initialized_workers]
                raise ValueError(f"Circular dependency detected. Remaining workers: {remaining}")

    def _build_event_routes(self) -> None:
        """Build the event routing map from config.

        Each worker in the config has an "events" list containing "worker_name": "event_name" pairs
        that specify which events from which workers it wants to receive.
        """
        # Don't add stop events to routing - they're handled directly
        for worker_name, worker_config in self.config.items():
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
                    raise ValueError(f"Event spec for {worker_name} should have exactly one key:value pair: {event_spec}")

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
        # Find UI worker if it exists
        ui_worker = None
        for name, worker in self.workers.items():
            if isinstance(worker, Ui):
                ui_worker = (name, worker)
                break

        # Start non-UI workers first
        for name, worker in self.workers.items():
            if not isinstance(worker, Ui):
                self.logger.debug(f"Starting worker: {name}")
                worker.start(self)

        # Start UI worker in main thread if it exists
        if ui_worker is not None:
            name, worker = ui_worker
            self.logger.debug(f"Starting UI worker: {name}")
            worker.start(self, run_in_main_thread=True)

    def stop(self) -> None:
        """Stop all workers."""
        self.logger.debug("Stopping all workers")
        for name, worker in self.workers.items():
            self.logger.debug(f"Stopping worker: {name}")
            worker.stop()
        self.logger.debug("All workers stopped")

    def __call__(self, event: Event) -> None:
        """Handle an event by routing it to appropriate workers.

        Args:
            event: Event to handle
        """
        self.logger.debug(f"Controller received event: {event.event_name} from {event.worker_name}")

        # Handle stop event from any worker by stopping all workers
        if event.event_name == "stop":
            self.logger.debug(f"Controller received stop event from {event.worker_name}, stopping all workers")
            self.stop()
            return

        # Get target workers for this event type
        targets = self.event_routes.get(event.event_name, set())
        self.logger.debug(f"Routing event {event.event_name} to targets: {targets}")

        # Route event to each target worker
        for target in targets:
            target_lower = target.lower()
            if target_lower in self.workers:
                self.logger.debug(f"Controller routing event to worker: {target}")
                self.workers[target]._dispatch(event)
            else:
                self.logger.warning(f"Unknown target worker: {target}")

    def __enter__(self) -> "Controller":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()