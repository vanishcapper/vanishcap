"""Worker for controlling the drone using DJITelloPy."""

import time
from typing import Any, Dict, Optional

from djitellopy import Tello
from vanishcap.event import Event
from vanishcap.worker import Worker


class Drone(Worker):  # pylint: disable=too-many-instance-attributes
    """Worker that controls the drone using DJITelloPy."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the drone worker.

        Args:
            config: Configuration dictionary containing:
                - ip: IP address of the drone
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - max_speed: Maximum speed in cm/s (default: 50)
                - follow_distance: Distance to maintain from target in cm (default: 100)
                - movement_threshold: Minimum movement threshold in normalized coordinates [-1, 1] (default: 0.1)
                - offline: Whether to run in offline mode (default: false)
        """
        super().__init__("drone", config)

        # Configuration
        self.max_speed = config.get("max_speed", 50)
        self.follow_distance = config.get("follow_distance", 100)
        self.movement_threshold = config.get("movement_threshold", 0.1)  # Now in [-1, 1] range
        self.offline = config.get("offline", False)
        self.logger.debug(
            "Initialized drone worker with max_speed=%d, follow_distance=%d, movement_threshold=%.2f, offline=%s",
            self.max_speed,
            self.follow_distance,
            self.movement_threshold,
            self.offline,
        )

        # Initialize drone connection if not offline
        self.drone = Tello()
        if not self.offline:
            self.drone.connect()
            self.logger.info("Connected to drone at IP: %s", config.get("ip", "192.168.10.1"))
            self._dispatch_command("set_speed", self.max_speed)
        else:
            self.logger.warning("Running in offline mode - skipping drone connection")

        # State
        self.is_flying = False
        self.current_target: Optional[Dict[str, float]] = None
        self.last_target_time = 0.0
        self.target_timeout = 1.0  # seconds

    def _dispatch_command(self, command: str, *args: Any) -> None:
        """Dispatch a command to the drone.

        Args:
            command: Name of the command to dispatch
            *args: Arguments to pass to the command
        """
        self.logger.info("Dispatching command %s with args %s (offline mode: %s)", command, args, self.offline)
        if self.offline:
            return

        try:
            # Get the command method from the drone
            cmd_method = getattr(self.drone, command)
            # Call the command with the provided arguments
            cmd_method(*args)
            self.logger.debug("Successfully executed command %s", command)
        except AttributeError:
            self.logger.error("Unknown drone command: %s", command)
        except (ConnectionError, TimeoutError) as e:
            self.logger.error("Connection error dispatching command %s: %s", command, e)
        except ValueError as e:
            self.logger.error("Invalid argument for command %s: %s", command, e)
        except RuntimeError as e:
            self.logger.error("Runtime error dispatching command %s: %s", command, e)

    def _task(self) -> None:
        """Run one iteration of the drone control loop."""
        current_time = time.time()

        # Check if we have a valid target
        if self.current_target and (current_time - self.last_target_time) < self.target_timeout:
            self._follow_target()
        else:
            if self.current_target:
                self.logger.debug("Target lost - timeout after %.1f seconds", current_time - self.last_target_time)
            self.current_target = None
            if self.is_flying:
                self.logger.debug("No valid target - stopping movement")
                self._dispatch_command("send_rc_control", 0, 0, 0, 0)  # Stop movement

    def _follow_target(self) -> None:
        """Follow the current target using proportional control.

        The target position is now given in normalized coordinates [-1, 1]:
        - Positive x = target is right of center
        - Negative x = target is left of center
        - Positive y = target is below center
        - Negative y = target is above center
        """
        if not self.current_target:
            return

        # Get target position in normalized coordinates [-1, 1]
        norm_x = self.current_target["x"]
        norm_y = self.current_target["y"]
        confidence = self.current_target.get("confidence", 0.0)
        self.logger.debug(
            "Target position: (%.2f, %.2f), confidence: %.2f",
            norm_x,
            norm_y,
            confidence,
        )

        # Calculate movement commands using proportional control
        # Scale the normalized position by max_speed to get movement speed
        # Invert y-axis since positive y means target is below (drone should move forward)
        fb = int(-norm_y * self.max_speed)
        yaw = int(norm_x * self.max_speed)

        # Apply confidence-based scaling
        fb = int(fb * confidence)
        yaw = int(yaw * confidence)

        # Clamp values to max_speed
        fb = max(min(fb, self.max_speed), -self.max_speed)
        yaw = max(min(yaw, self.max_speed), -self.max_speed)

        # Only move if offset is significant
        if abs(norm_x) < self.movement_threshold:
            yaw = 0
        if abs(norm_y) < self.movement_threshold:
            fb = 0

        self.logger.debug(
            "Movement commands - fb: %d, yaw: %d (threshold: %.2f)",
            fb,
            yaw,
            self.movement_threshold,
        )
        # Send movement commands
        self._dispatch_command("send_rc_control", 0, fb, 0, yaw)

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: The event to handle
        """
        if event.event_name == "target":
            # Update target position
            self.current_target = event.data
            self.last_target_time = time.time()
            self.logger.debug("Received new target position: (%.2f, %.2f)", event.data["x"], event.data["y"])

            # Start flying if not already
            if not self.is_flying:
                self.logger.debug("Starting flight")
                self._dispatch_command("takeoff")
                self.is_flying = True

    def _finish(self) -> None:
        """Clean up resources."""
        if self.is_flying:
            self.logger.debug("Landing drone")
            self._dispatch_command("land")
        if not self.offline:
            self.logger.debug("Ending drone connection")
            self._dispatch_command("end")
