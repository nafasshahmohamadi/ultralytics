# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations # Added from new version

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List, Union # Added typing imports

import numpy as np
import torch # Added torch import

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """Create a function that converts input to n-tuple by repeating singleton values."""

    def parse(x):
        """Parse input to return n-tuple by repeating singleton values n times."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"] # Kept stricter formats from new version

__all__ = ("Bboxes", "Instances")  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes in multiple formats.

    Supports 'xyxy', 'xywh', 'ltwh'. Handles format conversion, scaling, area calculation.
    Input should be numpy arrays.
    """

    def __init__(self, bboxes: np.ndarray | torch.Tensor, format: str = "xyxy") -> None: # Default format changed to xyxy
        """
        Initialize Bboxes with data and format. Merged tensor handling from your version.

        Args:
            bboxes (np.ndarray | torch.Tensor): Array/Tensor of shape (N, 4) or (4,).
            format (str): Format ('xyxy', 'xywh', 'ltwh'). Default is 'xyxy'.
        """
        assert format in _formats, f"Invalid bounding box format: {format}, must be one of {_formats}"

        # --- Merged from your version: Handle Tensor input and ensure float32 ---
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()  # Convert tensor to numpy
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32)

        # Ensure bboxes is at least 2D
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        # --- End merge ---

        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4

        self.bboxes = bboxes.astype(np.float32) # Ensure float32 from your version
        self.format = format


    def convert(self, format: str) -> None:
        """Convert bounding box format."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else: # format == "ltwh"
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    def mul(self, scale: int | tuple | list) -> None:
        """Multiply bounding box coordinates by scale factor(s)."""
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: int | tuple | list) -> None:
        """Add offset to bounding box coordinates."""
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        """Return the number of bounding boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        """Concatenate a list of Bboxes objects."""
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty((0, 4)), format="xyxy") # Ensure shape is (0, 4)
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        # Added format consistency check from new version
        formats = {b.format for b in boxes_list}
        assert len(formats) == 1, "All Bboxes must have the same format for concatenation."
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis), format=formats.pop())


    def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes:
        """Retrieve bounding boxes using indexing."""
        if isinstance(index, int):
            # Keep original format
            return Bboxes(self.bboxes[index].reshape(1, -1), format=self.format)
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        # Keep original format
        return Bboxes(b, format=self.format)


class Instances:
    """Container for bounding boxes, segments, and keypoints."""

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: list | np.ndarray = None, # Allow list input for segments
        keypoints: np.ndarray = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """Initialize Instances with boxes, segments, keypoints, format, normalization."""
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized

        # --- Merged from your version: Robust segment initialization ---
        if segments is None:
            self.segments = np.array([])
        else:
            # Handle list of arrays or numpy array directly
            if isinstance(segments, (list, tuple)) and len(segments) > 0 and isinstance(segments[0], np.ndarray):
                 # Attempt to stack if possible, otherwise use object dtype
                 try:
                     self.segments = np.stack(segments)
                 except ValueError: # Ragged array
                     self.segments = np.array(segments, dtype=object)
            elif isinstance(segments, np.ndarray):
                 self.segments = segments
            elif len(segments) == 0:
                 self.segments = np.array([])
            else: # Fallback for unexpected types
                 self.segments = np.array(segments, dtype=object)

        # Ensure segments is always a numpy array, even if empty or object type
        if not isinstance(self.segments, np.ndarray):
             self.segments = np.array(self.segments if segments is not None else [], dtype=object)
        if self.segments.size == 0: # Standardize empty representation
             self.segments = np.empty((0, 0, 2), dtype=np.float32) # Shape (0, 0, 2) to signify empty segments
        # --- End merge ---


    def convert_bbox(self, format: str) -> None:
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    # --- Kept implementations from the new original version ---
    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        """Scale coordinates by given factors using Bboxes.mul."""
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        if self.segments.size > 0: # Check size instead of None
            self.segments[..., 0] *= scale_w
            self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w: int, h: int) -> None:
        """Convert normalized coordinates to absolute coordinates using Bboxes.mul."""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        if self.segments.size > 0:
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        """Convert absolute coordinates to normalized coordinates using Bboxes.mul."""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        if self.segments.size > 0:
            self.segments[..., 0] /= w
            self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        """Add padding to coordinates using Bboxes.add."""
        assert not self.normalized, "you should add padding with absolute coordinates."
        self._bboxes.add(offset=(padw, padh, padw, padh))
        if self.segments.size > 0:
            self.segments[..., 0] += padw
            self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh
    # --- End kept implementations ---

    def __getitem__(self, index: int | np.ndarray | slice) -> Instances:
        """Retrieve instances using indexing."""
        # Use .size check for segments
        segments = self.segments[index] if self.segments.size > 0 else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        # Use internal _bboxes for consistent format handling
        bboxes = self._bboxes[index]
        return Instances(
            bboxes=bboxes.bboxes, # Pass the np.ndarray
            segments=segments,
            keypoints=keypoints,
            bbox_format=bboxes.format, # Keep format from indexed Bboxes
            normalized=self.normalized,
        )

    def flipud(self, h: int) -> None:
        """Flip coordinates vertically."""
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else: # xywh or ltwh
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        if self.segments.size > 0:
            self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        """Flip coordinates horizontally."""
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else: # xywh or ltwh
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        if self.segments.size > 0:
            self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w: int, h: int) -> None:
        """Clip coordinates to stay within image boundaries."""
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)

        # Kept efficient clipping from new version
        if self.segments.size > 0:
             self.segments[..., 0] = self.segments[..., 0].clip(0, w)
             self.segments[..., 1] = self.segments[..., 1].clip(0, h)

        if self.keypoints is not None:
            # Set out of bounds visibility to zero
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def remove_zero_area_boxes(self) -> np.ndarray:
        """Remove zero-area boxes."""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            # Use .size check
            if self.segments.size > 0:
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes: np.ndarray, segments: np.ndarray = None, keypoints: np.ndarray = None):
        """Update instance variables."""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
             # Use robust initialization logic here too
             if isinstance(segments, (list, tuple)) and len(segments) > 0 and isinstance(segments[0], np.ndarray):
                 try:
                     self.segments = np.stack(segments)
                 except ValueError:
                     self.segments = np.array(segments, dtype=object)
             elif isinstance(segments, np.ndarray):
                 self.segments = segments
             elif len(segments) == 0:
                 self.segments = np.empty((0, 0, 2), dtype=np.float32)
             else:
                 self.segments = np.array(segments, dtype=object)
             # Ensure empty segments have consistent shape
             if self.segments.size == 0:
                  self.segments = np.empty((0, 0, 2), dtype=np.float32)
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self) -> int:
        """Return the number of instances."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis=0) -> Instances:
        """Concatenate a list of Instances objects."""
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
             # Return empty instance with default format and normalization
             return cls(np.empty((0, 4)), segments=np.empty((0,0,2)), keypoints=None, bbox_format="xyxy", normalized=True)
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        # Check format consistency across list
        formats = {ins._bboxes.format for ins in instances_list}
        assert len(formats) == 1, "All Instances must have the same bbox format for concatenation."
        bbox_format = formats.pop()
        # Check normalization consistency
        normalized_states = {ins.normalized for ins in instances_list}
        assert len(normalized_states) == 1, "All Instances must have the same normalization state for concatenation."
        normalized = normalized_states.pop()


        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)

        # Handle segments concatenation robustly
        all_segments = [ins.segments for ins in instances_list if ins.segments.size > 0]
        if not all_segments:
             cat_segments = np.empty((0, 0, 2), dtype=np.float32)
        else:
             # Check if segments are object arrays (ragged) or uniform
             is_ragged = any(seg.dtype == object for seg in all_segments) or \
                         len({seg.shape[1] for seg in all_segments if seg.ndim == 3}) > 1 # Check points dimension

             if is_ragged:
                  # If any are ragged, or shapes differ, concatenate as object array
                  cat_segments_list = []
                  for ins in instances_list:
                       if ins.segments.ndim > 1: # Handle potentially empty segments correctly
                            cat_segments_list.extend(list(ins.segments))
                  cat_segments = np.array(cat_segments_list, dtype=object) if cat_segments_list else np.empty((0,), dtype=object)

             else:
                  # If all are uniform and shapes match, try direct concatenation
                  try:
                       cat_segments = np.concatenate(all_segments, axis=axis)
                  except ValueError: # Fallback if direct concat fails (e.g., empty arrays mixed)
                       cat_segments_list = [seg for seg in all_segments if seg.size > 0]
                       cat_segments = np.concatenate(cat_segments_list, axis=axis) if cat_segments_list else np.empty((0, all_segments[0].shape[1] if all_segments else 0, 2), dtype=np.float32)

        # Standardize empty segment representation after concatenation
        if cat_segments.size == 0:
             cat_segments = np.empty((0, 0, 2), dtype=np.float32)


        cat_keypoints = np.concatenate([b.keypoints for b in instances_list if b.keypoints is not None], axis=axis) if use_keypoint else None # Ensure keypoints check works even if some are None
        # Handle case where all instances might have None keypoints initially
        if use_keypoint and cat_keypoints is not None and cat_keypoints.size == 0:
             cat_keypoints = None # Reset to None if concatenation resulted in empty array


        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self) -> np.ndarray:
        """Return bounding boxes."""
        return self._bboxes.bboxes
