"""Command line interface for veil."""

import click
from pathlib import Path

from veil.controller import Controller
from veil.event import Event


@click.group(invoke_without_command=True)
@click.argument("config", type=click.Path(exists=True))
def cli(config: str):
    """Run the veil pipeline with the specified config file.

    Args:
        config: Path to the YAML configuration file
    """
    # Create the controller with the config file
    controller = Controller(config)
    
    # Start the video worker
    controller(Event("video", "run", None)) 