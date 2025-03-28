"""Veil workers module."""

from veil.workers.detector import Detector
from veil.workers.navigator import Navigator
from veil.workers.ui import Ui
from veil.workers.video import Video

__all__ = ["Detector", "Navigator", "UI", "Video"]
