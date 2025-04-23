"""Base driver interface for drone control."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


# pylint: disable=too-many-instance-attributes
class BaseDroneDriver(ABC):
    """Abstract base class for drone drivers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the base driver.

        Args:
            config: Driver configuration dictionary
        """
        self.config = config
        self.max_linear_velocity = config.get("max_linear_velocity", 50.0)
        self.max_angular_velocity = config.get("max_angular_velocity", 50.0)
        self.max_vertical_velocity = config.get("max_vertical_velocity", 30.0)
        self.field_of_view = config.get("field_of_view", 82.6)
        self.disable_yaw = config.get("disable_yaw", False)
        self.disable_xy = config.get("disable_xy", False)
        self.disable_z = config.get("disable_z", False)
        self.logger = logging.getLogger(config.get("name"))

    @abstractmethod
    def connect(self) -> None:
        """Connect to the drone."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the drone."""

    @abstractmethod
    def takeoff(self) -> None:
        """Take off."""

    @abstractmethod
    def land(self) -> None:
        """Land."""

    @abstractmethod
    def _send_rc_control(self, left_right: int, forward_back: int, up_down: int, yaw: int) -> None:
        """Send RC control commands to the drone.

        Args:
            left_right: Left/right velocity [-100, 100]
            forward_back: Forward/backward velocity [-100, 100]
            up_down: Up/down velocity [-100, 100]
            yaw: Yaw velocity [-100, 100]
        """

    def send_rc_control(self, left_right: int, forward_back: int, up_down: int, yaw: int) -> None:
        """Send RC control commands to the drone, respecting disable flags.

        Args:
            left_right: Left/right velocity [-100, 100]
            forward_back: Forward/backward velocity [-100, 100]
            up_down: Up/down velocity [-100, 100]
            yaw: Yaw velocity [-100, 100]
        """
        # Apply disable flags
        if self.disable_xy:
            left_right = 0
            forward_back = 0
        if self.disable_z:
            up_down = 0
        if self.disable_yaw:
            yaw = 0

        self._send_rc_control(left_right, forward_back, up_down, yaw)

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the drone.

        Returns:
            Dict[str, Any]: Current state dictionary
        """

    @abstractmethod
    def streamon(self) -> None:
        """Start video streaming."""

    @abstractmethod
    def streamoff(self) -> None:
        """Stop video streaming."""

    def get_max_linear_velocity(self) -> float:
        """Get the maximum linear velocity.

        Returns:
            float: Maximum linear velocity in cm/s
        """
        return self.max_linear_velocity

    def get_max_angular_velocity(self) -> float:
        """Get the maximum angular velocity.

        Returns:
            float: Maximum angular velocity in deg/s
        """
        return self.max_angular_velocity

    def get_max_vertical_velocity(self) -> float:
        """Get the maximum vertical velocity.

        Returns:
            float: Maximum vertical velocity in cm/s
        """
        return self.max_vertical_velocity

    def get_field_of_view(self) -> float:
        """Get the camera field of view.

        Returns:
            float: Field of view in degrees
        """
        return self.field_of_view
