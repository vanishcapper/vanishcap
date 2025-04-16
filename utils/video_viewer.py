"""A video viewer application that displays video with frame number and time overlays.

This module provides a CLI tool to view video from various sources (camera, file, or stream)
with frame number and elapsed time overlays. It supports saving the output video with
the original frames.
"""

import logging
import time
from typing import Optional, Tuple

import click
import numpy as np
import pygame
from vidgear.gears import CamGear, WriteGear

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def handle_events() -> bool:
    """Handle pygame events and return whether to continue running.

    Returns:
        bool: True if the application should continue running, False if it should quit
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # type: ignore # pylint: disable=no-member
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:  # type: ignore # pylint: disable=no-member
            return False
    return True

def setup_video_stream(video_source: str) -> Tuple[CamGear, float, np.ndarray]:
    """Set up the video stream and get initial frame and FPS.

    Args:
        video_source: Path to video file, camera index, or stream URL

    Returns:
        Tuple of (stream, fps, first_frame)

    Raises:
        RuntimeError: If video source cannot be read
    """
    # Start video stream
    stream = CamGear(source=video_source).start()

    # Get first frame
    frame = stream.read()
    if frame is None:
        raise RuntimeError("Could not read video source")

    # Get video framerate
    fps = stream.framerate
    if fps <= 0:
        fps = 30  # Default to 30 FPS if framerate can't be determined
        logger.info("Could not determine video framerate, using default: 30 FPS")
    else:
        logger.info("Video framerate: %.2f FPS", fps)

    return stream, fps, frame

def display_frame(screen: pygame.Surface, frame: np.ndarray, frame_number: int, start_time: float) -> None:
    """Display a frame with overlays using pygame.

    Args:
        screen: The pygame surface to display on
        frame: The video frame to display
        frame_number: Current frame number
        start_time: Time when video playback started
    """
    # Convert frame to RGB (OpenCV uses BGR)
    frame_rgb = np.flip(frame, axis=2)

    # Transpose frame to correct orientation
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))

    # Create pygame surface from frame
    surface = pygame.surfarray.make_surface(frame_rgb)

    # Create font for overlays
    font = pygame.font.Font(None, 36)

    # Calculate time since start
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Create overlay text
    frame_text = font.render(f"Frame: {frame_number}", True, (255, 255, 255))
    time_text = font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))

    # Display frame
    screen.blit(surface, (0, 0))

    # Display overlays
    screen.blit(frame_text, (10, 10))
    screen.blit(time_text, (10, 50))

    pygame.display.flip()

@click.command()
@click.argument('video_source', type=click.UNPROCESSED)
@click.option('--save', '-s', 'output_path', type=click.Path(), help='Path to save the output video')
def main(video_source: str, output_path: Optional[str] = None) -> None:
    """Display video with frame number and time overlays.

    VIDEO_SOURCE can be:
    - An integer (e.g., 0) for camera index
    - A string path to a video file
    - A string URL for a streaming source
    """
    try:
        # Convert to int if possible (camera source)
        try:
            video_source = int(video_source)
        except ValueError:
            pass  # Keep as string for file path or URL

        # Initialize pygame
        pygame.init()  # type: ignore # pylint: disable=no-member

        # Set up video stream
        stream, fps, frame = setup_video_stream(video_source)
        frame_delay = int(1000 / fps)  # Convert to milliseconds

        # Create pygame window
        screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        pygame.display.set_caption("Video Viewer")

        # Initialize video writer if save path is provided
        writer = None
        if output_path:
            output_params = {
                "-input_framerate": fps,
                "-vcodec": "libx264",
                "-crf": 23,
                "-preset": "fast"
            }
            writer = WriteGear(output_filename=output_path, **output_params)
            logger.info("Saving video to: %s", output_path)

        # Initialize variables
        frame_number = 0
        start_time = time.time()
        running = True

        # Main loop
        while running:
            # Check for quit events
            running = handle_events()

            # Read frame
            frame = stream.read()
            if frame is None:
                break

            # Display frame with overlays
            display_frame(screen, frame, frame_number, start_time)

            # Write frame if saving is enabled
            if writer is not None:
                writer.write(frame)

            frame_number += 1

            # Control frame rate using video's native framerate
            pygame.time.delay(frame_delay)

        # Print summary after video ends
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("\nVideo playback summary:")
        logger.info("Total frames: %d", frame_number)
        logger.info("Total time: %.2f seconds", total_time)
        logger.info("Average FPS: %.2f", frame_number / total_time if total_time > 0 else 0)

    except (IOError, ValueError, RuntimeError) as e:
        logger.error("Error: %s", str(e))
    finally:
        # Cleanup
        if 'stream' in locals():
            stream.stop()
        if writer is not None:
            writer.close()
        pygame.quit()  # type: ignore # pylint: disable=no-member

if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
