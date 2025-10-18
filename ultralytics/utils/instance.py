# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations # Added from new version

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List, Union # Kept List, Union from your version

import numpy as np
import torch # Added torch import from your version

# Keep ops imports from both versions
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
_formats = ["xyxy", "xywh", "ltwh"] # Basic formats for conversion logic
# Extended formats including normalized and rotated variants for initialization check
_extended_formats = ["xyxy", "xywh", "ltwh", "xywhn", "xyxyn", "xywhr"]


__all__ = ("Bboxes", "Instances")  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes in multiple formats. Optimized for NumPy arrays.
    """

    def __init__(self, bboxes: Union[np.ndarray, torch.Tensor, list], format: str = "xyxy") -> None: # Default format changed to xyxy to match new version, allow Tensor/list input
        """
        Initialize the Bboxes class with bounding box data in a specified format.

        Args:
            bboxes (np.ndarray | torch.Tensor | list): Array/Tensor/List of bounding boxes. Expects shape (N, >=4) or (>=4,).
                                                        Supports standard 4-coordinate formats and 5-coordinate xywhr.
            format (str): Format of the bounding boxes, one of 'xyxy', 'xywh', 'ltwh', 'xywhn', 'xyxyn', 'xywhr'.
        """
        assert format in _extended_formats, f"unsupported format {format}"

        # Your robust input handling
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy() # Ensure tensor is moved to CPU before converting
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32)

        # Handle single box input, ensure 2D array
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)

        # Check minimum required dimensions based on format
        min_vals = 5 if format == "xywhr" else 4
        if bboxes.shape[0] > 0: # Only check shape if there are boxes
             assert bboxes.ndim == 2, f"Input bboxes must be 2D, but got {bboxes.ndim}D"
             assert bboxes.shape[1] >= min_vals, f"bboxes shape should be at least (N, {min_vals}) for format '{format}', but got {bboxes.shape}"

        self.bboxes = bboxes.astype(np.float32) # Ensure float32
        self.format = format

    def convert(self, format: str) -> None:
        """
        Convert bounding box format from one type to another (only supports core formats).

        Args:
            format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        assert format in _formats, f"Invalid target bounding box format: {format}, must be one of {_formats}"
        # Prevent conversion from/to extended formats implicitly
        assert self.format in _formats, f"Cannot convert from format '{self.format}'. Only {_formats} are supported for conversion."

        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else: # ltwh
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes[:, :4] = func(self.bboxes[:, :4]) # Apply conversion only to the first 4 coords
        self.format = format

    def areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes (ignores rotation for xywhr)."""
        if self.format == "xywhr":
            # Area calculation for rotated boxes might need specific logic if angle affects perceived area.
            # For simplicity, calculating based on w*h here.
             return self.bboxes[:, 2] * self.bboxes[:, 3]
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 2] * self.bboxes[:, 3]  # format xywh or ltwh (using w*h)
        )

    def mul(self, scale: int | tuple | list) -> None:
        """
        Multiply bounding box coordinates (first 4) by scale factor(s).

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates. If int, the same scale is applied to
                all coordinates.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: int | tuple | list) -> None:
        """
        Add offset to bounding box coordinates (first 4).

        Args:
            offset (int | tuple | list): Offset(s) for four coordinates. If int, the same offset is applied to
                all coordinates.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2] # Adjust index if needed for specific formats
        self.bboxes[:, 3] += offset[3] # Adjust index if needed for specific formats

    def __len__(self) -> int:
        """Return the number of bounding boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            # Return Bboxes with shape (0, 4) for consistency, assuming default format if empty
            return cls(np.empty((0, 4), dtype=np.float32), format="xyxy")
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]

        # Ensure all formats are the same before concatenating
        first_format = boxes_list[0].format
        assert all(b.format == first_format for b in boxes_list), "All Bboxes must have the same format to concatenate."

        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis), format=first_format)


    def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes:
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.
        """
        if isinstance(index, int):
            # Keep original format when indexing single item
            return Bboxes(self.bboxes[index].reshape(1, -1), format=self.format)
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        # Keep original format when slicing/indexing multiple items
        return Bboxes(b, format=self.format)


class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray | list = None, # Allow list input for segments
        keypoints: np.ndarray = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """
        Initialize the Instances object.

        Args:
            bboxes (np.ndarray): Bounding boxes, shape (N, 4) or (N, 5 for xywhr).
            segments (np.ndarray | list, optional): Segmentation masks, shape (N, M, 2) or list of arrays.
            keypoints (np.ndarray, optional): Keypoints, shape (N, K, D) e.g. (N, 17, 3).
            bbox_format (str): Format of bboxes. See `_extended_formats`.
            normalized (bool): Whether the coordinates are normalized [0, 1].
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints if keypoints is not None else np.empty((0, 0, 0)) # Ensure keypoints is always an array
        self.normalized = normalized

        # Your robust segment handling
        if segments is None:
            self.segments = np.empty((0, 0, 0)) # Standard empty shape
        elif isinstance(segments, list):
            # Handle list of arrays (potentially ragged)
             self.segments = np.array(segments, dtype=object) if segments else np.empty((0, 0, 0))
        elif isinstance(segments, np.ndarray):
             # Ensure correct dimensions if ndarray
             self.segments = segments if segments.size > 0 else np.empty((0, 0, 0))
        else:
             raise TypeError(f"Unsupported type for segments: {type(segments)}")


    def convert_bbox(self, format: str) -> None:
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    # Using newer scale, denormalize, normalize, add_padding methods
    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        """Scale coordinates by given factors."""
        scale_factor = (scale_w, scale_h, scale_w, scale_h)
        if self._bboxes.format == 'xywhr': # Handle 5-element xywhr scaling if needed
             scale_factor = (scale_w, scale_h, scale_w, scale_h, 1.0) # Assume angle is not scaled
        self._bboxes.mul(scale=scale_factor[:self.bboxes.shape[1]]) # Scale appropriate number of elements

        if bbox_only:
            return
        if self.segments.size > 0: # Check size instead of None
            self.segments[..., 0] *= scale_w
            self.segments[..., 1] *= scale_h
        if self.keypoints.size > 0: # Check size instead of None
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w: int, h: int) -> None:
        """Convert normalized coordinates to absolute coordinates."""
        if not self.normalized:
            return
        scale_factor = (w, h, w, h)
        if self._bboxes.format == 'xywhr':
             scale_factor = (w, h, w, h, 1.0) # Assume angle is not denormalized by w/h
        self._bboxes.mul(scale=scale_factor[:self.bboxes.shape[1]])

        if self.segments.size > 0:
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        if self.keypoints.size > 0:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        """Convert absolute coordinates to normalized coordinates."""
        if self.normalized:
            return
        scale_factor = (1 / w, 1 / h, 1 / w, 1 / h)
        if self._bboxes.format == 'xywhr':
             scale_factor = (1 / w, 1 / h, 1 / w, 1 / h, 1.0) # Assume angle is not normalized by w/h
        self._bboxes.mul(scale=scale_factor[:self.bboxes.shape[1]])

        if self.segments.size > 0:
            self.segments[..., 0] /= w
            self.segments[..., 1] /= h
        if self.keypoints.size > 0:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        """Add padding to coordinates."""
        assert not self.normalized, "you should add padding with absolute coordinates."
        # Use Bboxes.add method directly, assuming it correctly handles the format
        offset = (padw, padh, padw, padh)
        if self._bboxes.format == 'xywh': # Center coordinates need different padding logic
             offset = (padw, padh, 0, 0) # Only add to center x, y
        elif self._bboxes.format == 'ltwh': # Left-top coordinates need different padding logic
             offset = (padw, padh, 0, 0) # Only add to left, top
        elif self._bboxes.format == 'xywhr': # Center coordinates for rotated
             offset = (padw, padh, 0, 0, 0) # Only add to center x, y
        self._bboxes.add(offset=offset[:self.bboxes.shape[1]])

        if self.segments.size > 0:
            self.segments[..., 0] += padw
            self.segments[..., 1] += padh
        if self.keypoints.size > 0:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh


    def __getitem__(self, index: int | np.ndarray | slice) -> Instances:
        """Retrieve a specific instance or a set of instances using indexing."""
        segments = self.segments[index] if self.segments.size > 0 else self.segments
        keypoints = self.keypoints[index] if self.keypoints.size > 0 else self.keypoints
        # Use the Bboxes __getitem__ which preserves format
        bboxes_selection = self._bboxes[index]

        return Instances(
            bboxes=bboxes_selection.bboxes, # Pass the underlying numpy array
            segments=segments,
            keypoints=keypoints,
            bbox_format=bboxes_selection.format, # Use format from the selected Bboxes
            normalized=self.normalized,
        )

    def flipud(self, h: int) -> None:
        """Flip coordinates vertically."""
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else: # xywh, ltwh, xywhr (center y or top y)
            self.bboxes[:, 1] = h - self.bboxes[:, 1]

        if self.segments.size > 0:
             # Handle potential dtype=object for segments array
             if self.segments.dtype == object:
                 for i in range(len(self.segments)):
                     if self.segments[i] is not None and self.segments[i].size > 0:
                         self.segments[i][..., 1] = h - self.segments[i][..., 1]
             else:
                 self.segments[..., 1] = h - self.segments[..., 1]

        if self.keypoints.size > 0:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        """Flip coordinates horizontally."""
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else: # xywh, ltwh, xywhr (center x or left x)
            self.bboxes[:, 0] = w - self.bboxes[:, 0]

        if self.segments.size > 0:
            # Handle potential dtype=object for segments array
             if self.segments.dtype == object:
                 for i in range(len(self.segments)):
                     if self.segments[i] is not None and self.segments[i].size > 0:
                         self.segments[i][..., 0] = w - self.segments[i][..., 0]
             else:
                 self.segments[..., 0] = w - self.segments[..., 0]

        if self.keypoints.size > 0:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]


    def clip(self, w: int, h: int) -> None:
        """Clip coordinates to stay within image boundaries."""
        ori_format = self._bboxes.format
        # Clip only makes sense in xyxy format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        # Convert back if original format was different
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)

        # Your efficient segment clipping
        if self.segments.size > 0:
             # Handle potential dtype=object for segments array
             if self.segments.dtype == object:
                 for i in range(len(self.segments)):
                     if self.segments[i] is not None and self.segments[i].size > 0:
                         self.segments[i][..., 0] = self.segments[i][..., 0].clip(0, w)
                         self.segments[i][..., 1] = self.segments[i][..., 1].clip(0, h)
             else:
                 self.segments[..., 0] = self.segments[..., 0].clip(0, w)
                 self.segments[..., 1] = self.segments[..., 1].clip(0, h)

        if self.keypoints.size > 0:
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
        """
        Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.

        Returns:
            (np.ndarray): Boolean array indicating which boxes were kept.
        """
        good = self.bbox_areas > 1e-6 # Use small threshold instead of 0 for float comparison
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments.size > 0: # Check size
                self.segments = self.segments[good]
            if self.keypoints.size > 0: # Check size
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes: np.ndarray, segments: np.ndarray | list = None, keypoints: np.ndarray = None):
        """
        Update instance variables.

        Args:
            bboxes (np.ndarray): New bounding boxes.
            segments (np.ndarray | list, optional): New segments.
            keypoints (np.ndarray, optional): New keypoints.
        """
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        # Update segment handling
        if segments is not None:
             if isinstance(segments, list):
                 self.segments = np.array(segments, dtype=object) if segments else np.empty((0, 0, 0))
             elif isinstance(segments, np.ndarray):
                 self.segments = segments if segments.size > 0 else np.empty((0, 0, 0))
             else:
                 raise TypeError(f"Unsupported type for segments update: {type(segments)}")

        if keypoints is not None:
             self.keypoints = keypoints if keypoints.size > 0 else np.empty((0, 0, 0))


    def __len__(self) -> int:
        """Return the number of instances."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis=0) -> Instances:
        """
        Concatenate a list of Instances objects into a single Instances object.
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            # Return empty Instances with default/consistent properties
            return cls(np.empty((0, 4), dtype=np.float32), bbox_format="xyxy", normalized=True)
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        # Check consistency
        use_keypoint = instances_list[0].keypoints.size > 0
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized
        assert all(ins.keypoints.size > 0 == use_keypoint for ins in instances_list), "Keypoint presence mismatch."
        assert all(ins._bboxes.format == bbox_format for ins in instances_list), "Bounding box format mismatch."
        assert all(ins.normalized == normalized for ins in instances_list), "Normalization status mismatch."


        cat_boxes = Bboxes.concatenate([ins._bboxes for ins in instances_list], axis=axis)

        # Handle segments (more robustly checking size and dtype)
        all_segments = [ins.segments for ins in instances_list if ins.segments.size > 0]
        if not all_segments:
             cat_segments = np.empty((0, 0, 0))
        elif all(isinstance(s, np.ndarray) and s.ndim == 3 for s in all_segments): # Check if all are standard numpy arrays
             seg_len = [s.shape[1] for s in all_segments]
             if len(frozenset(seg_len)) > 1: # Resample needed
                 max_len = max(seg_len) if seg_len else 0
                 # Prepare list for concatenation, handling empty segments within instances
                 processed_segments = []
                 for ins in instances_list:
                     if ins.segments.size == 0:
                         processed_segments.append(np.zeros((0, max_len, 2), dtype=np.float32))
                     else:
                          # Assuming resample_segments works correctly with list of arrays
                          resampled = resample_segments(list(ins.segments), n=max_len)
                          # Ensure resampled is a stackable numpy array
                          processed_segments.append(np.stack(resampled, axis=0) if resampled else np.zeros((0, max_len, 2), dtype=np.float32) )

                 cat_segments = np.concatenate(processed_segments, axis=axis)

             else: # No resampling needed
                 cat_segments = np.concatenate(all_segments, axis=axis)
        else: # Handle list of arrays (dtype=object) or mixed case - may require careful handling or raise error
             # Simple concatenation for object arrays, assumes compatibility
             try:
                 cat_segments = np.concatenate([ins.segments for ins in instances_list], axis=axis)
             except ValueError: # Fallback or error if shapes truly mismatch in object array case
                 raise ValueError("Cannot concatenate segments of varying dimensions or types easily.")


        # Handle keypoints
        cat_keypoints = np.concatenate([ins.keypoints for ins in instances_list], axis=axis) if use_keypoint else None

        return cls(cat_boxes.bboxes, cat_segments, cat_keypoints, bbox_format, normalized)


    @property
    def bboxes(self) -> np.ndarray:
        """Return bounding boxes."""
        return self._bboxes.bboxes
