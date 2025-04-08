"""Utility script to measure Tello drone performance characteristics."""

import time
from typing import Optional

import click
from djitellopy import Tello


class DroneSpecs:
    """Class for measuring Tello drone performance characteristics."""

    def __init__(self) -> None:
        """Initialize the drone connection."""
        self.drone = Tello()
        self.drone.connect()
        print("Connected to drone")

    def measure_linear_velocity(self, distance: int, duration: float) -> float:
        """Measure the drone's linear velocity over a specified distance and duration.

        Args:
            distance: Distance to move in cm
            duration: Duration to move in seconds

        Returns:
            Measured velocity in cm/s
        """
        print(f"\nMeasuring linear velocity over {distance}cm in {duration}s")
        print("Taking off...")
        self.drone.takeoff()
        time.sleep(2)  # Wait for takeoff to complete

        # Move forward at max velocity
        print("Moving forward...")
        self.drone.send_rc_control(0, 100, 0, 0)
        time.sleep(duration)
        self.drone.send_rc_control(0, 0, 0, 0)  # Stop movement

        # Calculate velocity
        velocity = distance / duration
        print(f"Measured velocity: {velocity:.1f} cm/s")

        # Land
        print("Landing...")
        self.drone.land()
        time.sleep(2)  # Wait for landing to complete

        return velocity

    def measure_angular_velocity(self, yaw_rc: int) -> Optional[float]:
        """Measure the drone's angular velocity at a given RC command value.

        Args:
            yaw_rc: Yaw RC command value (-100 to 100)

        Returns:
            Measured angular velocity in deg/s, or None if measurement failed
        """
        print(f"\nMeasuring angular velocity with yaw_rc={yaw_rc}")
        print("Taking off...")
        self.drone.takeoff()
        time.sleep(2)  # Wait for takeoff to complete

        # First measure using rotate_clockwise and rotate_counter_clockwise
        print("\nMeasuring angular velocity using rotate_clockwise and rotate_counter_clockwise...")

        # Clockwise rotation
        print("\nRotating clockwise 360 degrees...")
        start_time = time.time()
        self.drone.rotate_clockwise(360)
        cw_duration = time.time() - start_time
        cw_velocity = 360 / cw_duration
        print(f"Clockwise rotation took {cw_duration:.2f} seconds")
        print(f"Clockwise angular velocity: {cw_velocity:.1f} deg/s")
        time.sleep(1)  # Brief pause between rotations

        # Counter-clockwise rotation
        print("\nRotating counter-clockwise 360 degrees...")
        start_time = time.time()
        self.drone.rotate_counter_clockwise(360)
        ccw_duration = time.time() - start_time
        ccw_velocity = 360 / ccw_duration
        print(f"Counter-clockwise rotation took {ccw_duration:.2f} seconds")
        print(f"Counter-clockwise angular velocity: {ccw_velocity:.1f} deg/s")

        # Use the average velocity to determine initial duration for send_rc_control
        avg_velocity = (cw_velocity + ccw_velocity) / 2
        duration = 360 / avg_velocity
        print(f"\nAverage angular velocity from rotate_?wise: {avg_velocity:.1f} deg/s")
        print(f"Will use {duration:.2f} seconds as initial duration for send_rc_control measurements")

        # Now measure using send_rc_control
        print("\nNow measuring with send_rc_control...")
        while True:
            print(f"\nRotating for {duration:.2f} seconds...")
            self.drone.send_rc_control(0, 0, 0, yaw_rc)
            time.sleep(duration)
            self.drone.send_rc_control(0, 0, 0, 0)  # Stop rotation

            # Get user input for actual rotation as proportion of 360 degrees
            while True:
                try:
                    proportion = float(input("Enter the rotation as a proportion of 360 degrees (e.g., 0.5 for 180°, 1.0 for 360°, 1.5 for 540°): "))
                    if proportion <= 0:
                        print("Please enter a positive number")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number")

            # Calculate actual angle and velocity
            angle = proportion * 360
            velocity = angle / duration
            print(f"Measured rotation: {angle:.1f} degrees")
            print(f"Measured angular velocity: {velocity:.1f} deg/s")

            # Calculate new duration to target 360 degrees
            new_duration = 360 / velocity
            print(f"Next rotation will be for {new_duration:.2f} seconds to target 360 degrees")

            # Ask if we should continue
            while True:
                response = input("Continue with next measurement? (y/n): ").lower()
                if response in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")

            if response == 'n':
                break

            duration = new_duration

        # Land
        print("Landing...")
        self.drone.land()
        time.sleep(2)  # Wait for landing to complete

        return velocity

    def cleanup(self) -> None:
        """Clean up drone connection."""
        self.drone.end()


@click.group()
def cli() -> None:
    """Tello drone performance measurement tool."""
    pass


@click.command()
@click.option("--distance", default=100, help="Distance to move in cm")
@click.option("--duration", default=1.0, help="Duration to move in seconds")
def linear_velocity(distance: int, duration: float) -> None:
    """Measure the drone's linear velocity."""
    specs = DroneSpecs()
    try:
        specs.measure_linear_velocity(distance, duration)
    finally:
        specs.cleanup()


@click.command()
@click.option("--yaw-rc", default=100, help="Yaw RC command value (-100 to 100)")
def angular_velocity(yaw_rc: int) -> None:
    """Measure the drone's angular velocity."""
    specs = DroneSpecs()
    try:
        specs.measure_angular_velocity(yaw_rc)
    finally:
        specs.cleanup()


cli.add_command(linear_velocity)
cli.add_command(angular_velocity)

if __name__ == "__main__":
    cli()