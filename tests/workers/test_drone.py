"""Unit tests for the Drone worker."""

# pylint: disable=protected-access,no-member

import time
import unittest
from unittest.mock import MagicMock, patch

from vanishcap.workers.drone import Drone
from vanishcap.event import Event


class TestDrone(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test cases for the Drone worker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock socket
        self.mock_socket = MagicMock()
        self.mock_socket.sendto.return_value = 0
        self.mock_socket.recvfrom.return_value = (b"ok", ("192.168.10.1", 8889))

        # Create mock Tello instance
        self.mock_tello = MagicMock()
        self.mock_tello.connect.return_value = None
        self.mock_tello.send_command_with_return.return_value = "ok"
        self.mock_tello.get_current_state.return_value = {
            "battery": 100,
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "height": 0,
            "tof": 0,
            "temp_low": 0,
            "temp_high": 0,
            "time": 0,
        }
        self.mock_tello.client_socket = self.mock_socket
        self.mock_tello.address = ("192.168.10.1", 8889)
        self.mock_tello.state_counter = 0

        # Create mock Tello class
        self.mock_tello_class = MagicMock(return_value=self.mock_tello)

        # Create mock UDP object
        self.mock_udp_object = MagicMock()
        self.mock_udp_object.responses = {"ok": "ok"}
        self.mock_udp_object.state = {
            "battery": 100,
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "height": 0,
            "tof": 0,
            "temp_low": 0,
            "temp_high": 0,
            "time": 0,
        }

        # Create patches
        self.patcher_socket = patch("socket.socket", return_value=self.mock_socket)
        self.patcher_tello = patch("vanishcap.workers.drone.Tello", self.mock_tello_class)
        self.patcher_tello_socket = patch("djitellopy.tello.socket.socket", return_value=self.mock_socket)
        self.patcher_drones = patch("djitellopy.tello.drones", {"192.168.10.1": self.mock_udp_object})
        self.patcher_threads = patch("djitellopy.tello.threads_initialized", True)
        self.patcher_send_command = patch("djitellopy.tello.Tello.send_command_with_return", return_value="ok")
        self.patcher_connect = patch("djitellopy.tello.Tello.connect", return_value=None)

        # Start patches
        self.mock_socket_instance = self.patcher_socket.start()
        self.mock_tello_instance = self.patcher_tello.start()
        self.mock_tello_socket_instance = self.patcher_tello_socket.start()
        self.mock_drones_instance = self.patcher_drones.start()
        self.mock_threads_instance = self.patcher_threads.start()
        self.mock_send_command_instance = self.patcher_send_command.start()
        self.mock_connect_instance = self.patcher_connect.start()

        # Create test config
        self.config = {
            "ip": "192.168.10.1",
            "max_speed": 50,
            "follow_distance": 100,
            "movement_threshold": 0.10,
            "offline": False,
        }

        # Create mock Tello drone
        self.mock_drone = MagicMock()

        # Mock internal methods
        self.mock_drone.send_command_with_return.return_value = "ok"
        self.mock_drone.send_control_command.return_value = True
        self.mock_drone.get_current_state.return_value = self.mock_udp_object["state"]
        self.mock_drone.get_own_udp_object.return_value = self.mock_udp_object

        # Mock basic commands
        self.mock_drone.connect.return_value = None
        self.mock_drone.takeoff.return_value = None
        self.mock_drone.land.return_value = None
        self.mock_drone.end.return_value = None
        self.mock_drone.streamon.return_value = None
        self.mock_drone.send_rc_control.return_value = None

        # Create patch for Tello class
        self.patcher_tello = patch("djitellopy.Tello", return_value=self.mock_drone)
        self.mock_tello_class = self.patcher_tello.start()

        # Create patch for send_command_with_return method
        self.patcher_send_command = patch("djitellopy.Tello.send_command_with_return", return_value="ok")
        self.mock_send_command = self.patcher_send_command.start()

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_tello.stop()
        self.patcher_socket.stop()
        self.patcher_drones.stop()
        self.patcher_threads.stop()
        self.patcher_tello_socket.stop()
        self.patcher_send_command.stop()
        self.patcher_connect.stop()

    def test_initialization(self):
        """Test initialization of drone worker."""
        drone = Drone(self.config)
        self.assertEqual(drone.max_linear_velocity, 50)
        self.assertEqual(drone.max_angular_velocity, 50)
        self.assertEqual(drone.follow_distance, 100)
        self.assertEqual(drone.movement_threshold, 0.10)
        self.assertEqual(drone.offline, False)
        self.assertEqual(drone.is_flying, False)
        self.assertIsNone(drone.current_target)
        self.assertEqual(drone.target_timeout, 1.0)
        self.assertEqual(getattr(drone.drone, "streamon").call_count, 1)

    def test_initialization_offline(self):
        """Test initialization in offline mode."""
        config = {"ip": "192.168.10.1", "offline": True}
        drone = Drone(config)
        self.assertEqual(drone.offline, True)
        self.assertEqual(getattr(drone.drone, "connect").call_count, 0)
        self.assertEqual(getattr(drone.drone, "streamon").call_count, 0)

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        config = {"ip": "192.168.10.1"}
        drone = Drone(config)
        self.assertEqual(drone.max_linear_velocity, 50)
        self.assertEqual(drone.max_angular_velocity, 50)
        self.assertEqual(drone.follow_distance, 100)
        self.assertEqual(drone.movement_threshold, 0.10)
        self.assertEqual(drone.offline, False)
        self.assertEqual(getattr(drone.drone, "streamon").call_count, 1)

    def test_dispatch_command_online(self):
        """Test command dispatching in online mode."""
        drone = Drone(self.config)
        drone._dispatch_command("takeoff")
        self.assertEqual(getattr(drone.drone, "takeoff").call_count, 1)

    def test_dispatch_command_offline(self):
        """Test command dispatching in offline mode."""
        config = {"ip": "192.168.10.1", "offline": True}
        drone = Drone(config)
        drone._dispatch_command("takeoff")
        self.assertEqual(getattr(drone.drone, "takeoff").call_count, 0)

    def test_dispatch_command_error_handling(self):
        """Test error handling in command dispatching."""
        drone = Drone(self.config)
        drone._dispatch_command("unknown_command")
        drone._dispatch_command("send_rc_control", 0, 0, 0, 0)
        self.assertEqual(getattr(drone.drone, "send_rc_control").call_count, 1)

    def test_follow_target(self):
        """Test target following logic."""
        drone = Drone(self.config)
        drone.current_target = {
            "x": 0.4,
            "y": -0.4,
            "confidence": 1.0,
            "bbox": [0.3, 0.5, 0.5, 0.7],  # Add bounding box for vertical control
        }
        drone.current_target["processed"] = False  # Ensure target is not processed
        drone._follow_target()
        # Expected: fb_rc and yaw_rc are calculated and sent together
        # ud_rc is calculated based on bbox[1] (top_y) = 0.5 vs target_top_y = 0.0
        # The vertical velocity is doubled due to negation of y-coordinate
        getattr(drone.drone, "send_rc_control").assert_called_once_with(0, 20, 16, 100)
        self.assertTrue(drone.current_target["processed"])  # Ensure target is processed

    def test_follow_target_threshold(self):
        """Test movement threshold in target following."""
        drone = Drone(self.config)
        drone.current_target = {
            "x": 0.05,
            "y": 0.05,
            "confidence": 1.0,
            "bbox": [0.0, 0.5, 0.1, 0.6],  # Add bounding box for vertical control
        }
        drone._follow_target()
        # Forward/back and yaw are under threshold, but vertical movement is not
        # The vertical velocity is doubled due to negation of y-coordinate
        getattr(drone.drone, "send_rc_control").assert_called_once_with(0, 25, 0, 0)

    def test_target_timeout(self):
        """Test target timeout handling."""
        drone = Drone(self.config)
        drone.is_flying = True  # Set drone to flying state
        drone.is_stopped = False
        drone.current_target = {"x": 0.4, "y": -0.4, "confidence": 1.0}
        drone.last_target_time = time.time() - 2.0  # Simulate timeout
        drone._task()  # Run task loop
        self.assertIsNone(drone.current_target)  # Target should be cleared
        getattr(drone.drone, "send_rc_control").assert_called_once_with(0, 0, 0, 0)  # Should stop movement

    def test_event_handling(self):
        """Test event handling."""
        drone = Drone(self.config)
        target_event = Event(
            "navigator",
            "target",
            {
                "x": 0.4,
                "y": -0.4,
                "confidence": 1.0,
                "bbox": [0.3, 0.5, 0.5, 0.7],  # Add bbox data
            },
            frame_number=1,
        )
        drone._dispatch(target_event)
        drone._task()  # Process the event
        self.assertEqual(drone.current_target["x"], 0.4)
        self.assertEqual(drone.current_target["y"], -0.4)
        self.assertEqual(drone.current_target["confidence"], 1.0)
        self.assertEqual(drone.current_target["bbox"], [0.3, 0.5, 0.5, 0.7])

    def test_finish(self):
        """Test cleanup in finish method."""
        drone = Drone(self.config)
        drone.is_flying = True
        drone._finish()
        self.assertEqual(drone.is_flying, False)
        self.assertEqual(getattr(drone.drone, "land").call_count, 1)
        self.assertEqual(getattr(drone.drone, "end").call_count, 1)


if __name__ == "__main__":
    unittest.main()
