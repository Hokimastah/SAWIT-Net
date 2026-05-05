"""Dataset loaders for SAWIT-Net.

Supported formats:
1. CSV: a table with image path/id column and label column.
2. Folder: ImageFolder-style directory, e.g. root/class_name/image.jpg.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from .utils import IMAGE_EXTENSIONS, is_image_file, list_images


class LabelEncoderMixin:
    """Mutable global label map used across continual sessions."""

    label_map: Dict[str, int]

    def _encode_labels(self, raw_labels: Iterable[str]) -> np.ndarray:
        encoded = []
        for lbl in raw_labels:
            label_name = str(lbl)
            if label_name not in self.label_map:
                self.label_map[label_name] = len(self.label_map)
            encoded.append(self.label_map[label_name])
        return np.array(encoded, dtype=np.int64)


class DefaultImageTransform:
    """Lightweight transform equivalent to resize + tensor + ImageNet normalization.

    This avoids importing torchvision in the dataset module, so importing
    ``sawit_net`` remains lightweight.
    """

    def __init__(self, image_size: int = 112):
        self.image_size = int(image_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


def default_transform(image_size: int = 112):
    return DefaultImageTransform(image_size=image_size)


class SAWITCSVDataset(Dataset, LabelEncoderMixin):
    """CSV image dataset.

    Parameters
    ----------
    csv_file_or_df:
        Either a CSV path or a pandas DataFrame.
    image_root:
        Base directory for relative image paths.
    label_map:
        Shared mutable mapping from string labels to integer class IDs.
    image_col:
        Column containing relative/absolute image paths. The original older code
        used the column name ``id``.
    label_col:
        Column containing class names.
    """

    def __init__(
        self,
        csv_file_or_df: str | Path | pd.DataFrame,
        image_root: str | Path,
        label_map: Dict[str, int],
        image_col: str = "id",
        label_col: str = "label",
        image_size: int = 112,
        transform=None,
        allow_missing_images: bool = False,
    ):
        self.image_root = Path(image_root)
        self.label_map = label_map
        self.image_col = image_col
        self.label_col = label_col
        self.allow_missing_images = allow_missing_images
        self.transform = transform or default_transform(image_size)
        self.image_size = image_size

        if isinstance(csv_file_or_df, pd.DataFrame):
            self.data = csv_file_or_df.copy().reset_index(drop=True)
        else:
            self.data = pd.read_csv(csv_file_or_df).reset_index(drop=True)

        required = {self.image_col, self.label_col}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"CSV/DataFrame is missing required columns: {sorted(missing)}")

        self.image_paths = self.data[self.image_col].astype(str).values
        self.labels = self._encode_labels(self.data[self.label_col].astype(str).values)

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.image_root / path

    def __getitem__(self, idx: int):
        raw_path = str(self.image_paths[idx])
        image_path = self._resolve_path(raw_path)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as exc:
            if not self.allow_missing_images:
                raise FileNotFoundError(f"Cannot read image: {image_path}") from exc
            image = torch.zeros(3, self.image_size, self.image_size)

        return image, label, raw_path


class SAWITFolderDataset(Dataset, LabelEncoderMixin):
    """ImageFolder-style dataset.

    Directory example:

    data/base/cat/001.jpg
    data/base/dog/001.jpg
    """

    def __init__(
        self,
        root: str | Path,
        label_map: Dict[str, int],
        image_size: int = 112,
        transform=None,
        allow_missing_images: bool = False,
    ):
        self.root = Path(root)
        self.label_map = label_map
        self.image_size = image_size
        self.transform = transform or default_transform(image_size)
        self.allow_missing_images = allow_missing_images

        records = []
        for class_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            label = class_dir.name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and is_image_file(image_path):
                    records.append((str(image_path), label))

        if not records:
            raise ValueError(f"No images found under folder dataset root: {self.root}")

        self.image_paths = np.array([r[0] for r in records])
        raw_labels = [r[1] for r in records]
        self.labels = self._encode_labels(raw_labels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = Path(str(self.image_paths[idx]))
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as exc:
            if not self.allow_missing_images:
                raise FileNotFoundError(f"Cannot read image: {image_path}") from exc
            image = torch.zeros(3, self.image_size, self.image_size)
        return image, label, str(image_path)


def build_dataset(source, cfg, label_map: Dict[str, int]):
    if cfg.dataset_type.lower() == "csv":
        return SAWITCSVDataset(
            csv_file_or_df=source,
            image_root=cfg.image_root,
            label_map=label_map,
            image_col=cfg.image_col,
            label_col=cfg.label_col,
            image_size=cfg.image_size,
            allow_missing_images=cfg.allow_missing_images,
        )
    if cfg.dataset_type.lower() == "folder":
        return SAWITFolderDataset(
            root=source,
            label_map=label_map,
            image_size=cfg.image_size,
            allow_missing_images=cfg.allow_missing_images,
        )
    raise ValueError("dataset_type must be 'csv' or 'folder'.")
