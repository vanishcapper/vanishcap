#!/usr/bin/env python3
"""CLI tool to measure YOLO model performance on input sources."""

import time
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def process_frame(model: YOLO, frame: np.ndarray, device: str = "cpu", model_verbose: bool = False) -> tuple[float, list]:
    """Process a single frame and return inference time and detections.

    Args:
        model: YOLO model instance
        frame: Input frame as numpy array

    Returns:
        tuple[float, list]: Inference time in seconds and list of detections
    """
    start_time = time.perf_counter()
    results = model(frame, verbose=model_verbose, device=device)
    inference_time = time.perf_counter() - start_time

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections.append({
                'class_id': int(box.cls[0]),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist()
            })

    return inference_time, detections


def process_video(model: YOLO, source: str, max_frames: Optional[int] = None, device: str = "cpu", model_verbose: bool = False) -> dict:
    """Process video source and collect performance metrics.

    Args:
        model: YOLO model instance
        source: Video source (file path or webcam index)
        max_frames: Maximum number of frames to process (None for all)

    Returns:
        dict: Performance metrics
    """
    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    # Initialize metrics
    metrics = {
        'total_frames': 0,
        'total_time': 0.0,
        'inference_times': [],
        'detections_per_frame': [],
        'fps_history': []
    }

    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_count = 0

    warmed_up = False
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break

        if not warmed_up:
            _ = model(frame, device=device)
            warmed_up = True

        # Process frame
        inference_time, detections = process_frame(model, frame, device, model_verbose)

        # Update metrics
        metrics['total_frames'] += 1
        metrics['total_time'] += inference_time
        metrics['inference_times'].append(inference_time)
        metrics['detections_per_frame'].append(len(detections))

        # Calculate current FPS
        if len(metrics['inference_times']) > 1:
            current_fps = 1.0 / inference_time
            metrics['fps_history'].append(current_fps)

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Calculate final metrics
    if metrics['total_frames'] > 0:
        metrics['avg_fps'] = metrics['total_frames'] / metrics['total_time']
        metrics['avg_inference_time'] = np.mean(metrics['inference_times'])
        metrics['avg_detections'] = np.mean(metrics['detections_per_frame'])
        metrics['min_fps'] = min(metrics['fps_history']) if metrics['fps_history'] else 0
        metrics['max_fps'] = max(metrics['fps_history']) if metrics['fps_history'] else 0

    return metrics


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('source')
@click.option('--max-frames', type=int, help='Maximum number of frames to process')
@click.option('--output', type=click.Path(), help='Output file for metrics (JSON)')
@click.option('--device', type=str, help='Device to run the model on', default="cpu")
@click.option('--model-verbose', type=bool, help='Verbose level for the model', default=False)
def main(model_path: str, source: str, max_frames: Optional[int], output: Optional[str], device: str, model_verbose: bool):
    """Measure YOLO model performance on input source.

    MODEL_PATH: Path to YOLO model file (.pt, .onnx, or .engine)
    SOURCE: Video file path or webcam index (e.g., 0 for default webcam)
    """
    # Load model
    click.echo(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Process video
    click.echo(f"Processing source: {source}")
    metrics = process_video(model, source, max_frames, device, model_verbose)

    # Display results
    click.echo("\nPerformance Metrics:")
    click.echo(f"Total Frames Processed: {metrics['total_frames']}")
    click.echo(f"Average FPS: {metrics['avg_fps']:.2f}")
    click.echo(f"Min FPS: {metrics['min_fps']:.2f}")
    click.echo(f"Max FPS: {metrics['max_fps']:.2f}")
    click.echo(f"Average Inference Time: {metrics['avg_inference_time']*1000:.2f} ms")
    click.echo(f"Average Detections per Frame: {metrics['avg_detections']:.2f}")

    # Save results if output file specified
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"\nMetrics saved to {output}")


if __name__ == '__main__':
    main()