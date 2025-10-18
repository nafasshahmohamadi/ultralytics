# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple # Added Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    DefaultWindowing, # Added DefaultWindowing
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    DATASET_CACHE_VERSION, # Added DATASET_CACHE_VERSION back
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
# DATASET_CACHE_VERSION = "1.0.3" # Replaced by import above


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.
    This version contains modifications for handling .npy files and custom transforms.
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        self.use_labels = True # Added this line from your version
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)
        # Note: self.labels are now loaded by BaseDataset __init__

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """
        Cache dataset labels, check images and read shapes. This method integrates .npy handling.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        # Adjusted verify_image_label arguments to include custom logic if needed,
        # or create a separate verification step for .npy files if verify_image_label can't handle them.
        # For now, assuming verify_image_label works or the primary loading logic handles verification.

        pbar = TQDM(range(total), desc=desc, total=total)
        # Reusing your robust loading logic within the caching process
        for i in pbar:
            im_file = self.im_files[i]
            label_file = self.label_files[i]
            try:
                # Load image (.npy or standard formats)
                if Path(im_file).suffix.lower() == ".npy":
                    im = np.load(im_file)
                else:
                    # Use cv2 flags based on channels defined in BaseDataset __init__
                    im = cv2.imread(im_file, self.cv2_flag)

                if im is None:
                    raise ValueError(f"Unable to read image {im_file}")

                # Standardize data type AFTER reading
                im = im.astype(np.float32) # Your dtype standardization

                shape = im.shape[:2]  # height, width

                # Load labels
                if Path(label_file).is_file():
                    with open(label_file) as f:
                        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        l = np.array(l, dtype=np.float32)
                else:
                    nm += 1 # Increment missing count
                    l = np.zeros((0, 5 + nkpt * ndim), dtype=np.float32) # Empty label for background

                # Check labels
                if len(l):
                    nf += 1 # Increment found count
                    if self.use_keypoints:
                        l[:, 5:] = l[:, 5:].reshape(-1, nkpt, ndim)
                        # l[:, 5:] = l[:, 5:] / np.array(shape, dtype=np.float32)[::-1] # Normalization done later
                    if self.single_cls:
                        l[:, 0] = 0 # Force single class

                    # Filter labels based on included classes
                    if self.classes is not None:
                        include = l[:, 0].astype(int)
                        j = np.isin(include, self.classes)
                        l = l[j]
                else:
                    ne += 1 # Increment empty count

                # Prepare label dictionary (similar to your format)
                label_dict = {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": l[:, 0:1].astype(np.int64) if len(l) else np.zeros((0, 1), dtype=np.int64), # Your dtype fix
                    "bboxes": l[:, 1:5] if len(l) else np.zeros((0, 4), dtype=np.float32), # Your dtype fix
                    "segments": [], # Segments handled separately if needed
                    "keypoints": l[:, 5:].reshape(-1, nkpt, ndim) if self.use_keypoints and len(l) else None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }

                # Add segments if needed (adapted from original cache_labels verify loop)
                if self.use_segments:
                    segments = []
                    if Path(label_file).is_file():
                        try:
                            with open(label_file) as f:
                                s = [x.split() for x in f.read().strip().splitlines() if len(x)]
                                segments = np.array(s, dtype=np.float32)[:, 5:] # Assuming segments start after cls, xywh
                                if self.classes is not None and len(l): # Filter segments based on class filter `j` applied to `l`
                                    segments = segments[j]
                        except Exception as e:
                            nc += 1
                            msgs.append(f"{self.prefix}WARNING ⚠️ Error loading segments {label_file}: {e}")
                    label_dict["segments"] = segments

                x["labels"].append(label_dict)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

            except Exception as e:
                nc += 1
                msgs.append(f"{self.prefix}WARNING ⚠️ Ignoring corrupt image/label: {im_file}: {e}")
                continue # Skip appending this label
        pbar.close()


        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")

        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, total
        x["msgs"] = msgs  # warnings
        x["version"] = DATASET_CACHE_VERSION # Use imported version

        # Save cache
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x


    def get_labels(self) -> list[dict]:
        """
        Return dictionary of labels for YOLO training. This version uses the updated caching logic.

        Returns:
            (list[dict]): List of label dictionaries.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError, IndexError): # Added IndexError
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
             # Check for empty labels list before trying to access elements
             if nf > 0: # If images were found but no labels, issue a warning
                 LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. See {HELP_URL}")
             else: # If no images were found at all
                 raise RuntimeError(f"No images found in {self.img_path}. See {HELP_URL}")

        # Update image files list based on labels read from cache
        self.im_files = [lb["im_file"] for lb in labels]

        # Check if the dataset is all boxes or all segments (from new original)
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels if lb) # Added check for non-empty lb
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths)) if labels else (0, 0, 0) # Handle empty labels case
        if self.use_segments and len_segments == 0:
             LOGGER.warning(
                 f"WARNING ⚠️ No segments found in {cache_path}, 'task=segment' training may not work correctly. {HELP_URL}"
             )
        # Handle case where segments were expected but boxes were found, and vice-versa
        if self.use_segments and len_boxes and not len_segments:
             LOGGER.warning(
                 f"WARNING ⚠️ A segment dataset was expected but only boxes were found. Running `task=detect` instead."
                 f"{HELP_URL}"
             )
             self.use_segments = False
        if not self.use_segments and len_segments:
             LOGGER.warning(
                 "WARNING ⚠️ A detect dataset was expected but segments were found. Please select `task=segment`."
                 f"{HELP_URL}"
             )
        # Check for segments=True specified but no segments found (from new original)
        if self.use_segments and len_segments == 0:
            LOGGER.warning(
                f"WARNING ⚠️ No segments found in {cache_path}, `task=segment` training may not work correctly. {HELP_URL}"
            )

        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")

        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Builds and returns data augmentation transforms for the dataset."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0 # Added cutmix handling from new original
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            # Validation pipeline: DefaultWindowing + LetterBox
            transforms = Compose([
                DefaultWindowing(), # Your custom DefaultWindowing
                LetterBox(self.imgsz, auto=self.rect, stride=self.stride, scaleup=False) # Updated args from new original
                ])

        # Common Format transform for both train and val
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr = hyp.bgr if self.augment else 0.0, # Added bgr handling from new original
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0. (From new original)

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)


    def update_labels_info(self, label: dict) -> dict:
        """
        Update label format for different tasks. (From new original)

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = max(segment_resamples, max_len) # Updated logic from new original
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)

        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collates data samples into batches using the logic from the new original version.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        # Sort items in each batch dictionary by keys for consistent processing
        batch = [dict(sorted(b.items())) for b in batch]
        # Aggregate values for each key across all dictionaries in the batch
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        # Process aggregated values based on key type
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                # Stack image tensors along a new batch dimension
                value = torch.stack(value, 0)
            elif k == "visuals":
                # Pad sequences for variable length data like text features
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                 # Concatenate tensors for labels and bounding boxes along the first dimension
                 if isinstance(value[0], torch.Tensor):
                     value = torch.cat(value, 0)
                 else: # Handle cases like empty lists or non-tensor data gracefully
                     value = torch.tensor(np.concatenate(value, axis=0)) if any(isinstance(v, np.ndarray) and v.size > 0 for v in value) else torch.empty(0)


            new_batch[k] = value

        # Prepare batch indices for associating labels with their respective images
        batch_idx = []
        for i in range(len(batch)):
            if "cls" in batch[i] and isinstance(new_batch["cls"], torch.Tensor):
                # Ensure cls_i is a tensor before getting its shape
                cls_tensor = new_batch["cls"][new_batch["batch_idx"] == i] if "batch_idx" in new_batch else batch[i]["cls"]
                if isinstance(cls_tensor, torch.Tensor):
                     batch_idx.append(torch.full((cls_tensor.shape[0],), i))
                elif isinstance(cls_tensor, np.ndarray) and cls_tensor.size > 0:
                     batch_idx.append(torch.full((cls_tensor.shape[0],), i))


        # Concatenate batch indices if available and store in new_batch
        if batch_idx:
             new_batch["batch_idx"] = torch.cat(batch_idx, 0)


        return new_batch


# The rest of the classes (YOLOMultiModalDataset, GroundingDataset, etc.) are kept from the new original version
# without modification, assuming your changes were specific to YOLODataset's handling of .npy files and transforms.

class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        """
        Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """Get negative text samples based on frequency threshold."""
        if not category_freq:  # Check if category_freq is empty
             return [] # Return empty list if no categories
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """
    Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            max_samples (int): Maximum number of samples to load for text augmentation.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: list[dict[str, Any]]) -> None:
        """
        Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided
        labels matches the expected count for known datasets. It performs validation
        against a predefined set of datasets with known instance counts.

        Args:
            labels (list[dict[str, Any]]): List of label dictionaries, where each dictionary
                contains dataset annotations. Each label dict must have a 'bboxes' key with
                a numpy array or tensor containing bounding box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count
                for a recognized dataset.

        Note:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels if label and "bboxes" in label) # Added check
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": str(im_file), # Ensure im_file is string
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        x["version"] = DATASET_CACHE_VERSION # Use imported version
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (list[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError, IndexError): # Added IndexError
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
             if "texts" in label: # Check if 'texts' key exists
                 for text_list in label["texts"]:
                     for t in text_list:
                         t = t.strip()
                         category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """Get negative text samples based on frequency threshold."""
        if not category_freq: # Check if category_freq is empty
             return [] # Return empty list if no categories
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """
        Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Dataset class for image classification tasks extending torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return subset of data and targets corresponding to given indices.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """
        Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> list[tuple]:
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
