"""Controller for managing worker threads and event routing."""

# pylint: disable=too-many-nested-blocks

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
from omegaconf import OmegaConf
from vanishcap.event import Event
from vanishcap.worker import Worker
from vanishcap.utils.logging import get_worker_logger
from vanishcap.utils.wifi import WifiManager, WifiError


class InitializationError(Exception):
    """Exception raised when controller initialization fails."""


class Controller:
    """Controller for managing worker threads and event routing."""

    def __init__(self, config_path: str) -> None:  # pylint: disable=too-many-branches,too-many-statements
        """Initialize the controller for potentially multiple drone systems.

        Args:
            config_path: Path to the configuration file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid or structure is wrong
            WifiError: If WiFi initialization fails for any drone system
            InitializationError: If any other initialization fails
        """
        # Load the full configuration
        self.full_config = self._load_config(config_path)

        # Extract global controller settings
        controller_config = self.full_config.get("controller", {})
        log_level = controller_config.get("log_level", "WARNING")

        self.global_offline_mode = controller_config.get("offline", False)
        self.logger = get_worker_logger("controller", log_level, controller_config.get("log_file", None))
        self.logger.warning("Initializing controller with config: %s", config_path)

        # Initialize storage for workers, routes, WiFi managers, and raw configs
        self.workers: Dict[str, Worker] = {}
        self.event_routes: Dict[Tuple[str, str], Set[str]] = {}  # (source_worker, event_name) -> set of target workers
        self.wifi_managers: Dict[str, WifiManager] = {}
        self.all_worker_configs: List[Dict[str, Any]] = []
        ui_config = self.full_config.get("ui", {})
        ui_disabled = ui_config.get("disabled", False)

        # Process each top-level section in the config
        for section_name, section_config in self.full_config.items():
            if section_name == "controller":
                continue  # Already processed

            if section_name == "ui":
                if not isinstance(section_config, dict):
                    raise ValueError(f"UI config section '{section_name}' must be a dictionary")
                if not ui_disabled:
                    if "name" not in section_config:  # Ensure UI has a name if enabled
                        raise ValueError("Enabled UI worker must have a 'name' field defined.")
                    self.all_worker_configs.append(section_config)
                continue  # Processed UI section

            # Assume other sections are drone systems
            self.logger.info("Processing drone system: %s", section_name)

            if not isinstance(section_config, dict):
                self.logger.warning("Skipping non-dictionary top-level section: %s", section_name)
                continue

            # Initialize WiFi for this drone system if not in global offline mode
            if not self.global_offline_mode:  # pylint: disable=too-many-nested-blocks,too-many-branches
                wifi_config = section_config.get("wifi")
                if wifi_config and isinstance(wifi_config, dict):
                    self.logger.info("Initializing WiFi for %s", section_name)
                    try:
                        # Pass controller's log level to WifiManager for consistency
                        wifi_config["log_level"] = log_level
                        wifi_manager = WifiManager(wifi_config)
                        self.wifi_managers[section_name] = wifi_manager

                        if wifi_manager.connect(
                            wifi_config["connect"]["ssid"], wifi_config["connect"].get("password", "")
                        ):
                            self.logger.info("Successfully connected to WiFi for %s", section_name)
                        else:
                            self.logger.error("Failed to connect to WiFi for %s", section_name)

                    except WifiError as e:
                        self.logger.error("WiFi initialization failed for %s: %s", section_name, e)
                        # Depending on requirements, we might continue or raise here.
                        # Let's raise for now to indicate a critical setup failure.
                        raise InitializationError(f"WiFi initialization failed for {section_name}: {e}") from e
                    except Exception as e:  # Catch other potential errors during WifiManager init/get_ip
                        self.logger.error(
                            "Unexpected error during WiFi setup for %s: %s", section_name, e, exc_info=True
                        )
                        raise InitializationError(f"WiFi setup failed for {section_name}: {e}") from e
                else:
                    self.logger.warning("No valid WiFi config found for %s, skipping WiFi setup.", section_name)
            else:
                self.logger.warning("Global offline mode enabled, skipping WiFi for %s", section_name)

            # Extract worker configurations for this drone system
            workers_list = section_config.get("workers")
            if isinstance(workers_list, list):
                for worker_entry in workers_list:
                    if isinstance(worker_entry, dict) and len(worker_entry) == 1:
                        worker_type, worker_config = next(iter(worker_entry.items()))
                        if isinstance(worker_config, dict):
                            # Add the inferred type to the config if not present
                            if "type" not in worker_config:
                                worker_config["type"] = worker_type
                            # Store the drone system name with the worker config
                            worker_config["_drone_system"] = section_name
                            self.all_worker_configs.append(worker_config)
                        else:
                            raise ValueError(f"Invalid worker config value for type '{worker_type}' in {section_name}")
                    else:
                        raise ValueError(
                            f"Invalid worker entry in {section_name}: {worker_entry}. Expected dict with one key."
                        )
            elif workers_list is not None:
                raise ValueError(f"'workers' key in {section_name} must be a list.")

        # Initialize OpenCV window if UI is not disabled
        if not ui_disabled:
            self.logger.info("Initializing OpenCV window for UI.")
            cv2.namedWindow("vanishcap", flags=cv2.WINDOW_GUI_NORMAL)  # pylint: disable=no-member
        else:
            self.logger.warning("UI worker is disabled, skipping OpenCV window creation.")

        # Initialize all collected workers in dependency order
        try:
            self._init_workers()  # Uses self.all_worker_configs implicitly now
        except Exception as e:
            self.logger.error("Failed to initialize workers: %s", e, exc_info=True)
            raise InitializationError(f"Failed to initialize workers: {e}") from e

        # Build event routing map based on all initialized workers
        try:
            self._build_event_routes()  # Uses self.all_worker_configs implicitly now
        except Exception as e:
            self.logger.error("Failed to build event routes: %s", e, exc_info=True)
            raise InitializationError(f"Failed to build event routes: {e}") from e

        self.logger.warning("Controller initialized successfully")

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

    def _can_init_worker(self, worker_config: Dict[str, Any], initialized_workers: Set[str]) -> bool:
        """Check if a worker's dependencies have been initialized.

        Args:
            worker_config: Configuration dictionary for the worker
            initialized_workers: Set of already initialized workers

        Returns:
            bool: True if all dependencies are initialized, False otherwise
        """
        dependencies = worker_config.get("depends_on", [])
        return all(dep in initialized_workers for dep in dependencies)

    def _init_workers(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Initialize all workers collected from the config in dependency order."""
        # Import the workers module
        worker_module = importlib.import_module("vanishcap.workers")

        # Validate collected worker configurations
        if not self.all_worker_configs:
            self.logger.warning("No worker configurations found to initialize.")
            return  # Nothing to do

        # Enforce that each worker has a name
        for config in self.all_worker_configs:
            if "name" not in config or not config["name"]:
                raise ValueError(f"All workers must have a non-empty 'name' field. Found issue in: {config}")

        # Enforce uniqueness of worker names
        worker_names = [config["name"] for config in self.all_worker_configs]
        if len(worker_names) != len(set(worker_names)):
            duplicate_names = {name for name in worker_names if worker_names.count(name) > 1}
            raise ValueError(f"Worker names must be unique. Found duplicates: {duplicate_names}")

        # Get all available worker classes from the workers module
        available_worker_classes = {
            name.lower(): cls
            for name, cls in inspect.getmembers(worker_module)
            if inspect.isclass(cls) and issubclass(cls, Worker) and cls != Worker
        }
        self.logger.debug("Available worker classes: %s", list(available_worker_classes.keys()))

        # Create a dictionary of worker configs keyed by name for easier lookup
        worker_configs_by_name = {config["name"]: config for config in self.all_worker_configs}

        # Initialize workers in dependency order
        initialized_workers: Set[str] = set()
        initialization_attempts = 0
        max_attempts = len(worker_configs_by_name) + 1  # Failsafe against infinite loops

        while len(initialized_workers) < len(worker_configs_by_name) and initialization_attempts < max_attempts:
            initialized_this_round = False
            for worker_name, worker_config in worker_configs_by_name.items():
                # Skip already initialized workers
                if worker_name in initialized_workers:
                    continue

                # Check if dependencies are satisfied
                if self._can_init_worker(worker_config, initialized_workers):
                    worker_type = worker_config.get("type", worker_name.lower())  # Type might be explicit or inferred
                    if worker_type not in available_worker_classes:
                        available = ", ".join(sorted(available_worker_classes.keys()))
                        raise ValueError(
                            f"Unknown worker type '{worker_type}' for worker '{worker_name}'. Available: {available}"
                        )

                    worker_class = available_worker_classes[worker_type]
                    self.logger.info("Initializing worker '%s' of type '%s'", worker_name, worker_type)
                    try:
                        # For drone workers, inject the WiFi interface from the drone system config
                        if worker_type == "drone":  # pylint: disable=too-many-nested-blocks,too-many-branches
                            drone_system = worker_config.get("_drone_system")
                            if drone_system:
                                system_config = self.full_config.get(drone_system, {})
                                wifi_config = system_config.get("wifi", {}).get("connect", {})
                                interface = wifi_config.get("interface")
                                if interface:
                                    if "driver" not in worker_config:
                                        worker_config["driver"] = {}
                                    worker_config["driver"]["interface"] = interface
                                    self.logger.info(
                                        "Injected WiFi interface '%s' into drone worker '%s' config",
                                        interface,
                                        worker_name,
                                    )

                        # Initialize the worker
                        self.workers[worker_name] = worker_class(worker_config)
                        initialized_workers.add(worker_name)
                        initialized_this_round = True
                        self.logger.debug("Successfully initialized worker: %s", worker_name)
                    except Exception as e:
                        self.logger.error("Failed to initialize worker '%s': %s", worker_name, e, exc_info=True)
                        raise InitializationError(f"Failed during initialization of worker '{worker_name}': {e}") from e

            if not initialized_this_round and len(initialized_workers) < len(worker_configs_by_name):
                # If we couldn't initialize any workers in a round, check for circular dependencies or missing deps
                uninitialized = set(worker_configs_by_name.keys()) - initialized_workers
                error_message = "Could not initialize all workers. Check for missing or circular dependencies among: "
                details = []
                for name in uninitialized:
                    deps = worker_configs_by_name[name].get("depends_on", [])
                    missing_deps = [dep for dep in deps if dep not in initialized_workers]
                    details.append(f"  - {name} (depends on: {deps}, missing: {missing_deps})")
                self.logger.error("%s%s", error_message, str(uninitialized))
                for detail in details:
                    self.logger.error(detail)
                raise ValueError(error_message + str(uninitialized) + "\nDetails:\n" + "\n".join(details))

            initialization_attempts += 1

        if len(initialized_workers) != len(worker_configs_by_name):
            # This should theoretically be caught by the circular dependency check, but acts as a failsafe
            raise InitializationError(
                f"Initialization loop finished, but not all workers were initialized. "
                f"Expected {len(worker_configs_by_name)}, got {len(initialized_workers)}."
            )

        self.logger.info("Initialized %d workers: %s", len(self.workers), list(self.workers.keys()))

    def _build_event_routes(self) -> None:
        """Build the event routing map from the collected worker configurations."""
        self.event_routes.clear()  # Start with empty routes

        # Iterate through the raw config list used for initialization
        for worker_config in self.all_worker_configs:
            worker_name = worker_config.get("name")
            if not worker_name:
                # This should have been caught in _init_workers, but double-check
                self.logger.warning("Skipping event route building for config without name: %s", worker_config)
                continue

            # Ensure the worker was actually initialized (might not be if disabled or error)
            if worker_name not in self.workers:
                self.logger.debug("Worker '%s' not found in initialized workers, skipping event routes.", worker_name)
                continue

            # Get the events list for this worker
            events = worker_config.get("events", [])
            if not isinstance(events, list):
                raise ValueError(f"Invalid events list for worker '{worker_name}': {events}. Must be a list.")

            # Add routes for each event this worker wants to receive
            for event_spec in events:
                if not isinstance(event_spec, dict):
                    raise ValueError(f"Invalid event spec for {worker_name}: {event_spec}. Expected dict.")

                # Each event spec should have a single key:value pair like {source_worker: event_name}
                if len(event_spec) != 1:
                    raise ValueError(f"Event spec for {worker_name} must have exactly one key:value pair: {event_spec}")

                source_worker_name, event_name = next(iter(event_spec.items()))

                # Validate source worker exists (was initialized)
                if source_worker_name not in self.workers:
                    # Check if the source worker config exists but wasn't initialized (e.g. disabled)
                    source_exists_in_config = any(c.get("name") == source_worker_name for c in self.all_worker_configs)
                    if source_exists_in_config:
                        self.logger.warning(
                            "Source worker '%s' in event spec for '%s' exists in config "
                            "but was not initialized (maybe disabled?). Skipping route.",
                            source_worker_name,
                            worker_name,
                        )
                        continue
                    raise ValueError(
                        f"Unknown source worker '{source_worker_name}' in event spec for '{worker_name}'. "
                        f"Available workers: {list(self.workers.keys())}"
                    )

                # Add the route: (source_worker, event_name) -> set of target worker names
                route_key = (source_worker_name, event_name)
                if route_key not in self.event_routes:
                    self.event_routes[route_key] = set()
                self.event_routes[route_key].add(worker_name)
                self.logger.debug(
                    "Added route: Event '%s' from '%s' -> '%s'", event_name, source_worker_name, worker_name
                )

        self.logger.info(
            "Built %d event routes for %d event types.",
            sum(len(targets) for targets in self.event_routes.values()),
            len(self.event_routes),
        )
        for (source, event), targets in self.event_routes.items():
            self.logger.debug("  Route: %s:%s -> %s", source, event, targets)

    def start(self) -> None:
        """Start all workers."""
        from vanishcap.workers.ui import Ui  # pylint: disable=import-outside-toplevel

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

        # Get target workers for this event type from this source
        route_key = (event.worker_name, event.event_name)
        targets = self.event_routes.get(route_key, set())
        self.logger.debug("Routing event %s from %s to targets: %s", event.event_name, event.worker_name, targets)

        # Route event to each target worker
        for target in targets:
            if target in self.workers:
                self.logger.debug("Controller routing event to worker: %s", target)
                self.workers[target]._dispatch(event)
            else:
                self.logger.warning("Unknown target worker: %s", target)

    def __enter__(self) -> "Controller":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: Stop workers and restore WiFi."""
        self.stop()

        # Restore previous WiFi connections if needed
        if not self.global_offline_mode and self.wifi_managers:
            self.logger.info("Restoring previous WiFi connections for all managed systems...")
            for system_name, manager in self.wifi_managers.items():
                try:
                    self.logger.info("Restoring connection for %s...", system_name)
                    manager.reconnect_previous()
                except Exception as e:  # pylint: disable=broad-except
                    # Using broad exception during cleanup is justified
                    self.logger.error("Failed to restore WiFi for %s: %s", system_name, e, exc_info=True)
        elif self.global_offline_mode:
            self.logger.info("Global offline mode enabled, skipping WiFi restoration.")
        else:
            self.logger.info("No WiFi managers were initialized, skipping WiFi restoration.")

        # Close OpenCV window if it was created
        if not self.full_config.get("ui", {}).get("disabled", False):
            self.logger.info("Closing OpenCV window.")
            cv2.destroyAllWindows()  # pylint: disable=no-member
