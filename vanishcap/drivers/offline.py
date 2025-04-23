"""Offline driver implementation for simulated mode."""

from typing import Dict, Any

from vanishcap.drivers.base import BaseDroneDriver


class OfflineDriver(BaseDroneDriver):
    """Driver implementation that simulates drone behavior for offline/testing mode."""

    def connect(self) -> None:
        """Simulate connection."""
        self.logger.debug("Would connect to drone at %s", self.config.get("ip", "192.168.10.1"))

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self.logger.debug("Would disconnect from drone")

    def takeoff(self) -> None:
        """Simulate takeoff."""
        self.logger.debug("Would take off")

    def land(self) -> None:
        """Simulate landing."""
        self.logger.debug("Would land")

    def _send_rc_control(self, left_right: int, forward_back: int, up_down: int, yaw: int) -> None:
        """Simulate sending RC control commands.

        Args:
            left_right: Left/right velocity [-100, 100]
            forward_back: Forward/backward velocity [-100, 100]
            up_down: Up/down velocity [-100, 100]
            yaw: Yaw velocity [-100, 100]
        """
        self.logger.debug(
            "Would send RC control: left_right=%d, forward_back=%d, up_down=%d, yaw=%d",
            left_right,
            forward_back,
            up_down,
            yaw,
        )

    def get_current_state(self) -> Dict[str, Any]:
        """Return simulated state.

        Returns:
            Empty state dictionary for now
        """
        self.logger.debug("Would get current drone state")
        return {"offline": True}

    def streamon(self) -> None:
        """Simulate starting video stream."""
        self.logger.debug("Would start video stream")

    def streamoff(self) -> None:
        """Simulate stopping video stream."""
        self.logger.debug("Would stop video stream")
