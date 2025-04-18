"""Worker for controlling the drone using DJITelloPy."""

import time
from typing import Any, Dict, Optional

from djitellopy import Tello
from vanishcap.worker import Worker


class Drone(Worker):  # pylint: disable=too-many-instance-attributes
    """Worker that controls the drone using DJITelloPy."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the drone worker.

        Args:
            config: Configuration dictionary containing:
                - ip: IP address of the drone
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - max_linear_velocity: Maximum linear velocity in cm/s (default: 50)
                - max_angular_velocity: Maximum angular velocity in deg/s (default: 50)
                - max_vertical_velocity: Maximum vertical velocity in cm/s (default: 30)
                - follow_distance: Distance to maintain from target in cm (default: 100)
                - movement_threshold: Minimum movement threshold in normalized coordinates [-1, 1] (default: 0.1)
                - field_of_view: Width of camera field of view in degrees (default: 82.6)
                - auto_takeoff: Whether to take off automatically without target (default: false)
                - offline: Whether to run in offline mode (default: false)
                - percent_angle_to_command: Percentage of target angle to rotate in each yaw command
                  [0, 100] (default: 100)
                - disable_yaw: Whether to disable yaw rotation (default: false)
                - disable_xy: Whether to disable forward/backward and left/right movement (default: false)
                - disable_z: Whether to disable up/down movement (default: false)
        """
        super().__init__("drone", config)

        # Configuration
        self.max_linear_velocity = config.get("max_linear_velocity", 50)
        self.max_angular_velocity = config.get("max_angular_velocity", 50)
        self.max_vertical_velocity = config.get("max_vertical_velocity", 30)
        self.follow_distance = config.get("follow_distance", 100)
        self.movement_threshold = config.get("movement_threshold", 0.1)  # Now in [-1, 1] range
        self.field_of_view = config.get("field_of_view", 82.6)  # FOV in degrees
        self.auto_takeoff = config.get("auto_takeoff", False)  # Take off without target
        self.offline = config.get("offline", False)
        self.percent_angle_to_command = config.get("percent_angle_to_command", 100)  # Default to 100%
        self.disable_yaw = config.get("disable_yaw", False)
        self.disable_xy = config.get("disable_xy", False)
        self.disable_z = config.get("disable_z", False)
        self.logger.debug(
            "Initialized drone worker with max_linear_velocity=%d, max_angular_velocity=%d, max_vertical_velocity=%d, "
            "follow_distance=%d, movement_threshold=%.2f, field_of_view=%.1f, auto_takeoff=%s, offline=%s, "
            "percent_angle_to_command=%d%%, disable_yaw=%s, disable_xy=%s, disable_z=%s",
            self.max_linear_velocity,
            self.max_angular_velocity,
            self.max_vertical_velocity,
            self.follow_distance,
            self.movement_threshold,
            self.field_of_view,
            self.auto_takeoff,
            self.offline,
            self.percent_angle_to_command,
            self.disable_yaw,
            self.disable_xy,
            self.disable_z,
        )

        # Initialize drone connection if not offline
        self.drone = Tello()
        if not self.offline:
            self.drone.connect()
            self.logger.info("Connected to drone at IP: %s", config.get("ip", "192.168.10.1"))
            self._dispatch_command("streamon")
        else:
            self.logger.warning("Running in offline mode - skipping drone connection")

        # State
        self.is_flying = False
        self.is_stopped = True  # Track if we're currently stopped
        self.ready_to_process_targets = False  # Track if we're ready to process targets
        self.current_target: Optional[Dict[str, float]] = None
        self.last_target_time = 0.0
        self.target_timeout = 1.0  # seconds
        self.yaw_start_time = 0.0
        self.yaw_duration = 0.0
        self.executing_yaw = False

    def stop_movement(self) -> None:
        """Stop all drone movement by sending zero RC commands."""
        if not self.is_stopped:
            self._dispatch_command("send_rc_control", 0, 0, 0, 0)
            self.is_stopped = True
            self.logger.debug("Drone stopped")

    def _normalize_velocity(self, velocity: float, max_velocity: float) -> int:
        """Convert a real-world velocity to a normalized RC command value.

        Args:
            velocity: Real-world velocity in appropriate units (cm/s or deg/s)
            max_velocity: Maximum velocity in the same units

        Returns:
            Normalized RC command value in range [-100, 100]
        """
        # Calculate normalized value in range [-1, 1]
        normalized = velocity / max_velocity

        # Convert to RC command range [-100, 100]
        rc_value = int(normalized * 100)

        # Clamp to valid range
        return max(min(rc_value, 100), -100)

    def _dispatch_command(self, command: str, *args: Any) -> None:
        """Dispatch a command to the drone.

        Args:
            command: Name of the command to dispatch
            *args: Arguments to pass to the command
        """
        self.logger.info("Dispatching command %s with args %s", command, args)

        # Handle state changes before executing commands
        if command == "land":
            self.ready_to_process_targets = False
            self.logger.debug("Preparing to land - no longer ready to process targets")
        elif command == "takeoff":
            self.ready_to_process_targets = False

        try:
            # Get the command method from the drone
            cmd_method = getattr(self.drone, command)
            # Call the command with the provided arguments
            if self.offline:
                if command == "takeoff":
                    time.sleep(3)
            else:
                cmd_method(*args)
            self.logger.debug("Successfully executed command %s", command)

            # Handle state changes for specific commands
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

    def _task(self) -> None:
        """Run one iteration of the drone control loop."""
        current_time = time.time()

        # Get latest events and find the latest target event
        latest_target_event = self._get_latest_events_and_clear().get("target", None)

        # Process the latest target event if found
        if latest_target_event is not None:
            # If we're executing a yaw and receive a new target, reset yaw execution
            if self.executing_yaw:
                self.executing_yaw = False
                self.logger.debug("New target detected - resetting yaw rotation")

            # Update target position from the latest event
            self.current_target = latest_target_event.data
            self.current_target["processed"] = False  # Initialize processed flag
            self.current_target["frame_number"] = latest_target_event.frame_number  # Store frame number
            self.last_target_time = current_time  # Update last seen time
            self.logger.debug(
                "Received new target position (frame %d): (%.2f, %.2f), current target: %s",
                latest_target_event.frame_number,
                self.current_target["x"],
                self.current_target["y"],
                self.current_target,
            )

            # Start flying if not already
            if not self.is_flying and not self.auto_takeoff:
                self.logger.debug("Target detected - taking off")
                self._dispatch_command("takeoff")
                self.is_flying = True  # State change occurs after successful command dispatch

        # Check if we should auto takeoff (only if no target found yet)
        elif self.auto_takeoff and not self.is_flying:
            self.logger.info("Auto takeoff enabled - taking off")
            self._dispatch_command("takeoff")
            self.is_flying = True
            self.stop_movement()

        # Check if we have a valid target and are ready to process it
        if self.current_target and (current_time - self.last_target_time) < self.target_timeout:
            if not self.current_target.get("processed", False):
                frame_number = self.current_target.get("frame_number")
                if frame_number is not None:
                    self.logger.info(
                        "Processing frame %d with target at (%.2f, %.2f)",
                        frame_number,
                        self.current_target["x"],
                        self.current_target["y"],
                    )
                if self.ready_to_process_targets:
                    self._follow_target()
                else:
                    self.logger.debug("Not ready to process targets - skipping follow_target")
        else:
            self.current_target = None

            # Stop movement and reset yaw execution if target is lost
            if self.executing_yaw:
                self.executing_yaw = False
                self.logger.debug("Target lost - stopping yaw rotation")

            self.logger.debug("No valid target - stopping movement")
            self.stop_movement()

        # Check if current yaw command has completed
        if self.executing_yaw and current_time >= self.yaw_start_time + self.yaw_duration:
            self.executing_yaw = False
            self.logger.debug("Completed yaw rotation - stopping yaw")
            self.stop_movement()

    def _calculate_yaw_duration(self, norm_x: float) -> float:
        """Calculate the duration needed for yaw rotation to center the target.

        Args:
            norm_x: Normalized x position of target [-1, 1]

        Returns:
            Duration in seconds needed for the yaw command
        """
        # Calculate the target's angular offset in degrees
        # norm_x range is [-1, 1], so multiply by half the FOV to get degrees
        angular_offset = norm_x * (self.field_of_view / 2)

        # Scale the angular offset by the percentage we want to rotate
        # Convert percentage to proportion by dividing by 100
        scaled_angular_offset = angular_offset * (self.percent_angle_to_command / 100.0)

        # Calculate how long we need to rotate at max_angular_velocity to cover this angle
        # Using absolute value since we care about magnitude, not direction
        duration = abs(scaled_angular_offset) / self.max_angular_velocity

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
        """Follow the current target using proportional control.

        The target position is now given in normalized coordinates [-1, 1]:
        - Positive x = target is right of center
        - Negative x = target is left of center
        - Positive y = target is below center
        - Negative y = target is above center
        """
        # Get target position in normalized coordinates [-1, 1]
        norm_x = self.current_target["x"]
        norm_y = self.current_target["y"]
        confidence = self.current_target.get("confidence", 0.0)

        # Get bounding box coordinates
        bbox = self.current_target.get("bbox", [0, 0, 0, 0])  # [x1, y1, x2, y2] in normalized coordinates
        top_y = -bbox[1]  # y1 is the top of the bounding box

        self.logger.debug(
            "Target position: (%.2f, %.2f), top_y: %.2f, confidence: %.2f",
            norm_x,
            norm_y,
            top_y,
            confidence,
        )

        current_time = time.time()

        # Calculate forward/backward movement using proportional control
        # Scale the normalized position by max_velocity to get movement velocity
        # Invert y-axis since positive y means target is below (drone should move forward)
        fb_velocity = -norm_y * self.max_linear_velocity

        # Calculate up/down movement to keep the top of the target at y = 0 (center of frame)
        target_top_y = 0.5
        ud_velocity = (top_y - target_top_y) * self.max_vertical_velocity

        # Apply confidence-based scaling
        fb_velocity *= confidence
        ud_velocity *= confidence

        # Convert velocities to normalized RC command values
        fb_rc = self._normalize_velocity(fb_velocity, self.max_linear_velocity)
        ud_rc = self._normalize_velocity(ud_velocity, self.max_vertical_velocity)
        self.logger.debug(
            "up/down params: top_y=%.2f, target_top_y=%.2f, ud_velocity=%.2f, ud_rc=%.2f",
            top_y,
            target_top_y,
            ud_velocity,
            ud_rc,
        )

        # Only move if offset is significant
        if abs(norm_y) < self.movement_threshold:
            fb_rc = 0
        if abs(top_y - target_top_y) < self.movement_threshold:
            ud_rc = 0

        # Handle yaw movement with timing
        if abs(norm_x) >= self.movement_threshold:
            self.yaw_duration = self._calculate_yaw_duration(norm_x)
            self.yaw_start_time = current_time
            self.executing_yaw = True

            # Use max yaw velocity in the appropriate direction
            yaw_rc = 100 if norm_x > 0 else -100

            self.logger.info("Starting timed yaw rotation: duration=%.2fs, rc=%d", self.yaw_duration, yaw_rc)
            self._dispatch_command(
                "send_rc_control",
                0,
                fb_rc if not self.disable_xy else 0,
                ud_rc if not self.disable_z else 0,
                yaw_rc if not self.disable_yaw else 0,
            )
        else:
            # Just handle forward/backward and up/down movement
            self.logger.debug(
                "Movement command - fb: %d, ud: %d (threshold: %.2f)", fb_rc, ud_rc, self.movement_threshold
            )
            self._dispatch_command(
                "send_rc_control", 0, fb_rc if not self.disable_xy else 0, ud_rc if not self.disable_z else 0, 0
            )

        # Mark target as processed
        self.current_target["processed"] = True

    def _finish(self) -> None:
        """Clean up resources."""
        if self.is_flying:
            self.logger.debug("Landing drone")
            self._dispatch_command("land")
            self.is_flying = False
        self.logger.debug("Ending drone connection")
        self._dispatch_command("end")
