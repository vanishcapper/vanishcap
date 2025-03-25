"""Controller class for managing workers and their configurations."""

import inspect
from pathlib import Path
from typing import Dict, Type, TypeVar

from omegaconf import OmegaConf

from veil.event import Event
from veil.worker import Worker

T = TypeVar("T", bound=Worker)


class Controller:
    """Controller class that manages workers and their configurations."""

    def __init__(self, config_path: str) -> None:
        """Initialize the controller with a config file.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If a config key doesn't match a worker class name
        """
        self.workers: Dict[str, Worker] = {}
        self.event_routes: Dict[str, Dict[str, str]] = {}  # Maps event names to {target_worker: event_name}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration and create workers.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If a config key doesn't match a worker class name
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        conf = OmegaConf.load(path)
        config = OmegaConf.to_container(conf, resolve=True)
        # Import workers module and get all worker classes
        from veil import workers
        
        worker_classes = {
            name: cls for name, cls in inspect.getmembers(workers)
            if (inspect.isclass(cls) and 
                issubclass(cls, Worker) and 
                cls != Worker)  # Exclude the base Worker class
        }

        # First pass: create all workers
        for name, worker_config in config.items():
            # Find worker class name case-insensitively
            worker_class_name = next(
                (key for key in worker_classes.keys() if key.lower() == name.lower()),
                None
            )
            
            if worker_class_name is None:
                raise ValueError(
                    f"Config key '{name}' doesn't match any worker class. "
                    f"Available workers: {', '.join(worker_classes.keys())}"
                )
            
            worker_type = worker_classes[worker_class_name]
            self.workers[name] = worker_type(worker_config)

        # Second pass: build event routing map
        for name, worker_config in config.items():
            if "events" in worker_config:
                events = worker_config["events"]
                # Skip if events is just "run" - this means the worker handles its own event loop
                if events == "run":
                    continue
                
                # Handle list of event routes
                if isinstance(events, list):
                    for route in events:
                        if not isinstance(route, dict):
                            raise ValueError(f"Invalid event route: {route}")
                        for source_worker, event_name in route.items():
                            if source_worker not in self.workers:
                                raise ValueError(
                                    f"Event route from worker '{source_worker}' targets non-existent worker '{name}'"
                                )
                            if event_name not in self.event_routes:
                                self.event_routes[event_name] = {}
                            self.event_routes[event_name][name] = event_name

    def __call__(self, event: Event) -> Event:
        """Handle an event by passing it to the appropriate worker.

        Args:
            event: The event to handle

        Returns:
            Event: The response event from the worker

        Raises:
            KeyError: If no worker is found for the event's worker_name
        """
        if event.worker_name not in self.workers:
            raise KeyError(f"No worker found for name: {event.worker_name}")
        
        # Get response from the worker
        response = self.workers[event.worker_name](event)
        
        # Route the response event if there are any routes defined
        if response.event_name in self.event_routes:
            # Create events for all target workers
            for target_worker, event_name in self.event_routes[response.event_name].items():
                target_event = Event(
                    target_worker,
                    event_name,
                    response.data
                )
                # Recursively handle the event
                self(target_event)
        
        return response 