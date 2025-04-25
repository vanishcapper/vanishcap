"""Worker for controlling the drone using a driver interface."""

from copy import deepcopy
from dataclasses import dataclass
import importlib
import time
from typing import Any, Dict, Optional

from vanishcap.drivers.base import BaseDroneDriver
from vanishcap.event import Event
from vanishcap.worker import Worker


@dataclass
class CommandState:
    """Represents the current RC control state of the drone.

    This class stores the normalized RC control values for the drone's movement:
    - lr: Left/right movement (-100 to 100)
    - fb: Forward/backward movement (-100 to 100)
    - ud: Up/down movement (-100 to 100)
    - yaw: Yaw rotation (-100 to 100)

    All values default to 0, representing no movement.
    """

    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0

    def __eq__(self, other: object) -> bool:
        """Compare this CommandState with another object for equality.

        Args:
            other: The object to compare with. Can be None, a CommandState, or any other type.

        Returns:
            bool: True if other is a CommandState with identical values, False otherwise.
            Returns False if other is None or not a CommandState.
        """
        # raise NotImplementedError("CommandState equality is not implemented")
        if other is None:
            return False
        if not isinstance(other, CommandState):
            return False
        return self.lr == other.lr and self.fb == other.fb and self.ud == other.ud and self.yaw == other.yaw


class Drone(Worker):  # pylint: disable=too-many-instance-attributes
    """Worker that controls the drone using a driver interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the drone worker.

        Args:
            config: Configuration dictionary containing:
                - driver: Driver configuration dictionary containing:
                    - name: Name of the driver to use (default: "tello")
                    - ip: IP address of the drone (for Tello)
                    - max_linear_velocity: Maximum linear velocity in cm/s (default: 50)
                    - max_angular_velocity: Maximum angular velocity in deg/s (default: 50)
                    - max_vertical_velocity: Maximum vertical velocity in cm/s (default: 30)
                    - field_of_view: Camera field of view in degrees (default: 82.6)
                    - disable_yaw: Whether to disable yaw rotation (default: false)
                    - disable_xy: Whether to disable forward/backward and left/right movement (default: false)
                    - disable_z: Whether to disable up/down movement (default: false)
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - name: Name of the worker (required)
                - follow_distance: Distance to maintain from target in cm (default: 100)
                - movement_threshold: Minimum movement threshold in normalized coordinates [-1, 1] (default: 0.1)
                - delay_between_timed_yaws: Delay between timed yaws in seconds (default: 1.0)
                - auto_takeoff: Whether to take off automatically without target (default: false)
                - percent_angle_to_command: Percentage of target angle to rotate in each yaw command
                  [0, 100] (default: 100)
                - follow_target_height: Where on the target's height to center (default: 0.0)
                  0.0 means center the top of the target, 1.0 means center the bottom
                - follow_target_width: Target width as a proportion of frame width (default: 0.3)
                  Used to control forward/backward movement when target is fully in frame
        """
        super().__init__(config)

        # Get driver configuration
        driver_config = config.get("driver", {})
        driver_name = driver_config.get("name", "tello")

        # Dynamically import and initialize driver
        try:
            driver_module = importlib.import_module(f"vanishcap.drivers.{driver_name}")
            driver_class = getattr(driver_module, f"{driver_name.title()}Driver")
            self.driver: BaseDroneDriver = driver_class(driver_config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load driver '{driver_name}': {e}") from e

        # Configuration
        self.follow_distance = config.get("follow_distance", 100)
        self.movement_threshold = config.get("movement_threshold", 0.1)
        self.delay_between_timed_yaws = config.get("delay_between_timed_yaws", 1.0)
        self.auto_takeoff = config.get("auto_takeoff", False)
        self.percent_angle_to_command = config.get("percent_angle_to_command", 100)
        self.follow_target_height = config.get("follow_target_height", 0.85)
        self.follow_target_width = config.get("follow_target_width", 0.3)

        # Initialize drone connection
        self.driver.connect()
        self.logger.info("Connected to drone using %s driver", driver_name)
        self.driver.streamon()

        # State
        self.current_command = CommandState()
        self.last_command = None
        self.is_flying = False
        self.ready_to_process_targets = False
        self.current_target: Optional[Dict[str, Any]] = None
        self.last_target_x: Optional[float] = None
        self.yaw_start_time = 0.0
        self.yaw_duration = 0.0
        self.commanded_yaw = 0
        self.executing_yaw = False
        self.searching_for_target = False
        self.search_start_time = 0.0

    def update_movement(self):
        """Update the drone's movement state and send RC commands to the driver.

        This method:
        1. Compares the current command state with the last command state
        2. If they differ, updates the last command state and sends new RC commands
        3. Skips sending commands if the current state is identical to the last state

        The method ensures that identical commands are not sent repeatedly to the drone,
        which helps prevent unnecessary communication and potential command queuing issues.
        """
        if self.last_command == self.current_command:
            self.logger.debug("Movement command is the same as the last command - skipping")
            return
        self.last_command = self.current_command
        self.driver.send_rc_control(
            self.current_command.lr, self.current_command.fb, self.current_command.ud, self.current_command.yaw
        )

    def _normalize_velocity(self, velocity: float, max_velocity: float) -> int:
        """Convert a real-world velocity to a normalized RC command value.

        Args:
            velocity: Real-world velocity in appropriate units (cm/s or deg/s)
            max_velocity: Maximum velocity in the same units

        Returns:
            Normalized RC command value in range [-100, 100]
        """
        normalized = velocity / max_velocity
        rc_value = int(normalized * 100)
        return max(min(rc_value, 100), -100)

    def _dispatch_command(self, command: str, *args: Any) -> None:
        """Dispatch a command to the drone driver.

        Args:
            command: Name of the command to dispatch
            *args: Arguments to pass to the command
        """
        self.logger.info("Dispatching command %s with args %s", command, args)

        if command == "land":
            self.ready_to_process_targets = False
            self.logger.debug("Preparing to land - no longer ready to process targets")
        elif command == "takeoff":
            self.ready_to_process_targets = False

        try:
            cmd_method = getattr(self.driver, command)
            cmd_method(*args)
            self.logger.debug("Successfully executed command %s", command)

            if command == "takeoff":
                self.ready_to_process_targets = True
                self.logger.debug("Takeoff successful - ready to process targets")
        except AttributeError:
            self.logger.error("Unknown drone command: %s", command)
        except (ConnectionError, TimeoutError) as e:
            self.logger.error("Connection error dispatching command %s: %s", command, e)
        except ValueError as e:
            self.logger.error("Invalid argument for command %s: %s", command, e)
        except RuntimeError as e:
            self.logger.error("Runtime error dispatching command %s: %s", command, e)

    def _process_target_event(self, latest_target_event: Event) -> None:
        """Process a new target event.

        Args:
            latest_target_event: The latest target event to process
        """
        if latest_target_event.data is None:
            self.logger.debug("Received empty target event - clearing current target")
            if self.current_target is not None:
                self.last_target_x = self.current_target["x"]
                self.searching_for_target = True
                self.search_start_time = time.time()
            self.current_target = None
            return

        self.current_target = latest_target_event.data
        self.current_target["processed"] = False
        self.current_target["frame_number"] = latest_target_event.frame_number
        self.current_target["timestamp"] = latest_target_event.timestamp
        self.last_target_x = self.current_target["x"]
        self.searching_for_target = False
        self.logger.debug(
            "Received new target position (frame %d): (%.2f, %.2f), bbox: (%f, %f, %f, %f)",
            latest_target_event.frame_number,
            self.current_target["x"],
            self.current_target["y"],
            self.current_target["bbox"][0],
            self.current_target["bbox"][1],
            self.current_target["bbox"][2],
            self.current_target["bbox"][3],
        )

        if not self.is_flying and not self.auto_takeoff:
            self.logger.debug("Target detected - taking off")
            self._dispatch_command("takeoff")
            self.is_flying = True

    def _handle_auto_takeoff(self) -> None:
        """Handle auto takeoff if enabled and no target is found."""
        if self.auto_takeoff and not self.is_flying:
            self.logger.info("Auto takeoff enabled - taking off")
            self._dispatch_command("takeoff")
            self.is_flying = True

    def _process_current_target(self) -> None:
        """Process the current target if valid.

        Args:
            current_time: Current time in seconds
        """
        if self.current_target:
            if not self.current_target["processed"] and self.ready_to_process_targets:
                self.logger.info(
                    "Processing frame %d with target at (%.2f, %.2f)",
                    self.current_target["frame_number"],
                    self.current_target["x"],
                    self.current_target["y"],
                )
                self._follow_target()
            elif not self.ready_to_process_targets:
                self.logger.debug("Not ready to process targets - skipping follow_target")
        else:
            if self.executing_yaw:
                self.executing_yaw = False
            if self.searching_for_target and self.last_target_x is not None:
                current_time = time.time()
                if current_time - self.search_start_time < 5.0:  # Search for 5 seconds
                    self.logger.debug("Searching for target in last known direction")
                    yaw_rc = 50 if self.last_target_x > 0 else -50  # Half speed yaw
                    self.current_command = CommandState(yaw=yaw_rc)
                    self.update_movement()
                else:
                    self.logger.debug("Search timeout - stopping yaw")
                    self.searching_for_target = False
                    self.current_command = CommandState()
                    self.update_movement()
            else:
                self.logger.debug("No valid target - stopping movement")
                self.current_command = CommandState()
                self.update_movement()

    def _task(self) -> None:
        """Run one iteration of the drone control loop."""
        state = self.driver.get_current_state()
        self.logger.debug("Current drone state: %s", state)

        current_time = time.time()
        latest_target_event = self._get_latest_events_and_clear().get("target", None)

        if latest_target_event is not None:
            self._process_target_event(latest_target_event)
        else:
            self._handle_auto_takeoff()

        self._process_current_target()

        if self.executing_yaw and current_time >= self.yaw_start_time + self.yaw_duration:
            self.executing_yaw = False
            self.last_command = deepcopy(self.current_command)
            self.current_command.yaw = 0
            self.logger.debug("Completed yaw rotation - stopping yaw")

        self.update_movement()

    def _calculate_yaw_duration(self, norm_x: float) -> float:
        """Calculate the duration needed for yaw rotation to center the target.

        Args:
            norm_x: Normalized x position of target [-1, 1]

        Returns:
            Duration in seconds needed for the yaw command
        """
        angular_offset = norm_x * (self.driver.get_field_of_view() / 4)
        scaled_angular_offset = angular_offset * (self.percent_angle_to_command / 100.0)
        duration = abs(scaled_angular_offset) / self.driver.get_max_angular_velocity()

        self.logger.debug(
            "Angular offset: %.2f degrees, scaled offset: %.2f degrees "
            "(%.1f%% of target), required yaw duration: %.2f s",
            angular_offset,
            scaled_angular_offset,
            self.percent_angle_to_command,
            duration,
        )

        return duration

    def _follow_target(self) -> None:
        """Follow the current target using proportional control."""
        target_pos = {
            "x": self.current_target["x"],
            "y": self.current_target["y"],
            "confidence": self.current_target.get("confidence", 0.0),
        }

        bbox = self.current_target.get("bbox", [0, 0, 0, 0])
        target_height = bbox[3] - bbox[1]
        target_y = bbox[1] + (target_height * self.follow_target_height)
        target_width = (bbox[2] - bbox[0]) / 2

        self.logger.debug(
            "Target position: (%.2f, %.2f), target_y: %.2f, target_width: %.2f, confidence: %.2f",
            target_pos["x"],
            target_pos["y"],
            target_y,
            target_width,
            target_pos["confidence"],
        )

        current_time = time.time()

        fb_velocity = 0
        width_error = target_width - self.follow_target_width
        if abs(bbox[0]) < 0.95 and abs(bbox[2]) < 0.95:
            fb_velocity = -width_error * self.driver.get_max_linear_velocity()

        frame_center_y = 0.5
        # If target's top edge is above the threshold (bbox[3] > 0.95), move upward
        if bbox[3] > 0.95 and width_error / self.follow_target_width < -0.1:
            ud_velocity = self.driver.get_max_vertical_velocity() / 3
            fb_velocity = 0
        else:
            ud_velocity = (target_y - frame_center_y) * self.driver.get_max_vertical_velocity()

        fb_velocity *= target_pos["confidence"]
        ud_velocity *= target_pos["confidence"]

        lr_rc = 0  # Implement left/right movement later
        fb_rc = self._normalize_velocity(fb_velocity, self.driver.get_max_linear_velocity())
        ud_rc = self._normalize_velocity(ud_velocity, self.driver.get_max_vertical_velocity())
        yaw_rc = 0

        if abs(target_width - self.follow_target_width) < self.movement_threshold:
            fb_rc = 0
        # Only apply movement threshold for downward movement
        if bbox[3] <= 0.95 and abs(target_y - frame_center_y) < self.movement_threshold:
            ud_rc = 0

        if (
            abs(target_pos["x"]) >= self.movement_threshold
            and current_time - self.yaw_start_time > self.delay_between_timed_yaws
        ):
            self.yaw_duration = self._calculate_yaw_duration(target_pos["x"])
            self.yaw_start_time = current_time
            self.executing_yaw = True

            yaw_rc = 100 if target_pos["x"] > 0 else -100
            self.commanded_yaw = yaw_rc

            self.logger.info("Starting timed yaw rotation: duration=%.2fs, rc=%d", self.yaw_duration, yaw_rc)

        self.current_command = CommandState(lr=lr_rc, fb=fb_rc, ud=ud_rc, yaw=yaw_rc)
        self.current_target["processed"] = True

    def _finish(self) -> None:
        """Clean up resources."""
        if self.is_flying:
            self.logger.debug("Stopping all movement before landing")
            self.current_command = CommandState()
            self.update_movement()
            self.logger.debug("Landing drone")
            self._dispatch_command("land")
            self.is_flying = False
        self.logger.debug("Ending drone connection")
        self.driver.disconnect()
