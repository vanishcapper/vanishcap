# Vanishcap

Autonomous UAV software to act as a partner when you'd rather not be seen. See [here](https://mario.fandom.com/wiki/Vanish_Cap) for the origin story.

## Installation

To install in development mode:

```bash
pip install -e .
```

To install with development dependencies (ipython, pdbpp, black, pylint):

```bash
pip install -e ".[dev]"
```

## Usage

After installation, you can use the `vanishcap` command:

```bash
# Show help
vanishcap --help

# Run vanishcap with config
vanishcap conf/controller_config.yaml
```

## Configuration

The system is configured using a YAML file. Here are the available configuration parameters:

### Controller Configuration

The controller section manages the overall system and worker coordination:

```yaml
controller:
  log_level: INFO  # Log level for the controller (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  log_file: "/tmp/vanishcap.log"  # Optional: Path to log file
  offline: false  # Whether to run in offline mode (no WiFi management)
  wifi:
    connect:
      ssid: "t2"  # Optional: SSID to connect to before starting workers
      password: ""  # Optional: Password for the SSID
    reconnect: true  # Optional: Whether to reconnect to previous SSID after workers finish
```

### Worker Configurations

Each worker has its own configuration section. All workers support these common parameters:
- `log_level`: Log level for the worker (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `profile_window`: Time window in seconds for profiling task execution time
- `depends_on`: List of worker names that must start before this worker
- `events`: List of events to receive from other workers in format `worker_name: event_name`

#### Video Worker

Captures and processes video frames:

```yaml
video:
  source: "udp://@0.0.0.0:11111"  # Video source (URL, camera index, or file path)
  save_path: "/tmp/vanishcap_recording.mp4"  # Optional: Path to save recorded video
```

#### Detector Worker

Performs object detection using YOLOv5:

```yaml
detector:
  model: yolov5nu  # Base model path without extension
  backend: tensorrt  # Backend to use (pytorch, tensorrt, onnx)
  frame_skip: 1  # Number of frames to skip between detections
```

#### Navigator Worker

Processes detections and generates navigation commands:

```yaml
navigator:
  target_class: "person"  # Class name to track
```

#### Drone Worker

Controls the drone using a driver interface:

```yaml
drone:
  driver:
    name: "tello"  # Name of the driver to use
    ip: "192.168.10.1"  # IP address of the drone
    max_linear_velocity: 80  # Maximum linear velocity in cm/s
    max_angular_velocity: 90  # Maximum angular velocity in deg/s
    max_vertical_velocity: 30  # Maximum vertical velocity in cm/s
    max_yaw_to_command: 20  # Maximum yaw command value [-100, 100]
    field_of_view: 86.2  # Camera field of view in degrees
    disable_yaw: false  # Whether to disable yaw rotation
    disable_xy: false  # Whether to disable forward/backward and left/right movement
    disable_z: false  # Whether to disable up/down movement
  follow_distance: 100  # Distance to maintain from target in cm
  follow_target_width: 0.5  # Target width as proportion of frame width
  follow_target_height: 0.75  # Where on target's height to center (0.0 = top, 1.0 = bottom)
  movement_threshold: 0.1  # Minimum movement threshold in normalized coordinates [-1, 1]
  delay_between_timed_yaws: 0.0  # Delay between timed yaws in seconds
  percent_angle_to_command: 100  # Percentage of target angle to rotate in each yaw command [0, 100]
  auto_takeoff: true  # Whether to take off automatically without waiting for a target
```

#### UI Worker

Displays video feed and worker profiling information:

```yaml
ui:
  # No additional parameters beyond common worker parameters
```

## Development

### Code Formatting

The project uses Black for code formatting with a line length of 120 characters. To format your code:

```bash
black .
```

### Linting

Pylint is configured to work alongside Black. To run the linter:

```bash
pylint vanishcap
```
