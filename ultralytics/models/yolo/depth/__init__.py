# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MDE (Monocular Depth Estimation) module for YOLO models.

This module provides depth estimation capabilities for YOLO models, allowing
simultaneous object detection and depth estimation.
"""

from .mde_model import MDE
from .train import MDETrainer
from .val import MDEValidator

__all__ = ["MDE", "MDETrainer", "MDEValidator"]
