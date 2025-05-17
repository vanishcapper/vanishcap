"""CLI utility for controlling multiple drones over WiFi."""

import contextlib
from dataclasses import dataclass
import os
import socket
import sys
import time
import threading
from typing import Optional, Dict, List

import cv2
import click
import numpy as np

from vanishcap.utils.wifi import WifiManager
from vanishcap.utils.logging import get_worker_logger


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

@dataclass
class DroneConfig:
    """Configuration for a single drone."""
    interface: str
    ssid: str
    name: str
    video_port: int = 11111  # Default video port


class Drone:
    """Class to handle a single drone."""

    def __init__(self, config: DroneConfig) -> None:
        """Initialize the drone.

        Args:
            config: Drone configuration
        """
        self.logger = get_worker_logger(f"drone_{config.name}", "INFO")
        self.config = config
        self.socket = None
        self.video_capture = None
        self.running = False
        self.wifi_manager = None
        self.current_frame = None

    def connect(self) -> bool:
        """Connect to the drone's WiFi network.

        Returns:
            bool: True if connection successful, False otherwise
        """
        wifi_config = {
            "connect": {
                "ssid": self.config.ssid,
                "interface": self.config.interface
            }
        }
        self.wifi_manager = WifiManager(wifi_config)
        return self.wifi_manager.connect(self.config.ssid)

    def setup_socket(self) -> bool:
        """Set up the control socket.

        Returns:
            bool: True if socket setup successful, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Bind to the specified interface
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, self.config.interface.encode())

            # Bind to any local address on port 8889
            self.socket.bind(('', 8889))
            self.socket.settimeout(5)  # 5 second timeout
            return True
        except Exception as e:
            self.logger.error(f"Failed to set up socket: {e}")
            return False

    def send_command(self, command: str) -> Optional[str]:
        """Send a command to the drone and wait for response.

        Args:
            command: Command to send

        Returns:
            Optional[str]: Response from drone, or None if no response
        """
        if not self.socket:
            self.logger.error("Socket not set up")
            return None

        try:
            self.logger.info(f"Sending command: {command}")
            self.socket.sendto(command.encode(), ('192.168.10.1', 8889))
            response, _ = self.socket.recvfrom(1024)
            return response.decode().strip()
        except socket.timeout:
            self.logger.warning(f"Timeout waiting for response to command: {command}")
            return None
        except Exception as e:
            self.logger.error(f"Error sending command {command}: {e}")
            return None

    def start_video(self) -> None:
        """Start capturing video from the drone."""
        self.logger.info(f"Starting video stream for {self.config.name} on port {self.config.video_port}")
        with stdchannel_redirected(sys.stderr, os.devnull):
            self.video_capture = cv2.VideoCapture(f'udp://@0.0.0.0:{self.config.video_port}')
        if not self.video_capture.isOpened():
            self.logger.error("Failed to open video stream")
            return

        self.running = True
        while self.running:
            with stdchannel_redirected(sys.stderr, os.devnull):
                ret, frame = self.video_capture.read()
            if not ret:
                self.logger.warning("Failed to read frame from video stream")
                continue

            # Add name to the frame
            cv2.putText(frame, self.config.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.current_frame = frame

        self.video_capture.release()

    def run(self) -> None:
        """Run the drone control sequence."""
        if not self.connect():
            self.logger.error("Failed to connect to WiFi")
            return

        if not self.setup_socket():
            self.logger.error("Failed to set up socket")
            return

        # Send commands
        commands = ["command", "streamon"]
        for cmd in commands:
            response = self.send_command(cmd)
            if response:
                self.logger.info(f"Response to {cmd}: {response}")
            time.sleep(1)  # Wait between commands

        # Start video in a separate thread
        video_thread = threading.Thread(target=self.start_video)
        video_thread.start()

        try:
            # Keep thread alive while video is running
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
            video_thread.join()


class DroneControl:
    """Class to handle multiple drones."""

    def __init__(self, drone_configs: List[DroneConfig]) -> None:
        """Initialize the drone control.

        Args:
            drone_configs: List of drone configurations
        """
        self.logger = get_worker_logger("drone_control", "INFO")
        self.drones: Dict[str, Drone] = {}
        for config in drone_configs:
            self.drones[config.name] = Drone(config)
        self.running = False

    def display_tiled_video(self) -> None:
        """Display tiled video from all drones."""
        self.running = True
        while self.running:
            # Get all frames
            frames = []
            for drone in self.drones.values():
                if drone.current_frame is not None:
                    frames.append(drone.current_frame)

            if not frames:
                time.sleep(0.1)
                continue

            # Calculate grid dimensions
            num_frames = len(frames)
            grid_cols = int(num_frames ** 0.5) + (1 if num_frames % int(num_frames ** 0.5) else 0)
            grid_rows = (num_frames + grid_cols - 1) // grid_cols

            self.logger.debug(f"Displaying {num_frames} frames in a {grid_rows}x{grid_cols} grid")

            # Get frame dimensions
            height, width = frames[0].shape[:2]

            # Create tiled image
            tiled_image = np.zeros((height * grid_rows, width * grid_cols, 3), dtype=np.uint8)

            # Place frames in grid
            for i, frame in enumerate(frames):
                row = i // grid_cols
                col = i % grid_cols
                tiled_image[row*height:(row+1)*height, col*width:(col+1)*width] = frame

            # Display tiled image
            cv2.imshow('Drone Video Grid', tiled_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                self.running = False
                break

        cv2.destroyAllWindows()

    def run(self) -> None:
        """Run all drones."""
        # Start video capture threads
        video_threads = []
        for drone in self.drones.values():
            thread = threading.Thread(target=drone.run)
            thread.start()
            video_threads.append(thread)

        # Start display thread
        display_thread = threading.Thread(target=self.display_tiled_video)
        display_thread.start()

        try:
            # Keep main thread alive while running
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
        for drone in self.drones.values():
            drone.running = False
        for thread in video_threads:
            thread.join()
        display_thread.join()


@click.command()
@click.option('--drones', '-d', required=True, multiple=True,
              help='Drone configuration in format "name:interface:ssid[:video_port]" (e.g. "drone1:wlan0:TELLO-XXXXXX:11111")')
def main(drones: List[str]) -> None:
    """Control multiple drones over WiFi.

    This utility will:
    1. Connect to each specified WiFi network
    2. Open control sockets on port 8889
    3. Send 'command' and 'streamon' commands to each drone
    4. Display video streams from each drone
    5. Exit when ESC is pressed on any video window

    Example usage:
        python drone_control.py -d "drone1:wlan0:TELLO-XXXXXX:11111" -d "drone2:wlan1:TELLO-YYYYYY:11112"
    """
    drone_configs = []
    for drone_str in drones:
        try:
            parts = drone_str.split(':')
            if len(parts) == 3:
                name, interface, ssid = parts
                video_port = 11111  # Default port
            elif len(parts) == 4:
                name, interface, ssid, video_port = parts
                video_port = int(video_port)
            else:
                raise ValueError("Invalid number of parts")
            drone_configs.append(DroneConfig(interface=interface, ssid=ssid, name=name, video_port=video_port))
        except ValueError as e:
            click.echo(f"Invalid drone configuration: {drone_str}. Use format 'name:interface:ssid[:video_port]'")
            return

    control = DroneControl(drone_configs)
    control.run()


if __name__ == '__main__':
    main()