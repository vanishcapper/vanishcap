# Veil

A command line tool.

## Dependencies

This project uses several key libraries:
- `ultralytics`: For YOLO-based object detection and tracking
- `djitellopy`: For controlling DJI Tello drones
- `opencv-python`: For image processing and computer vision tasks
- `vidgear`: For efficient video handling and streaming
- `pygame`: For display and user interface

## Installation

To install in development mode:

```bash
pip install -e .
```

To install with development dependencies (ipython, pdbpp, black, pylint):

```bash
pip install -e ".[dev]"
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
pylint veil
```

## Usage

After installation, you can use the `veil` command:

```bash
# Show help
veil --help

# Run the hello command
veil hello
``` 