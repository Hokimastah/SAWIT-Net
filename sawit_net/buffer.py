"""Replay buffer with herding selection and class prototypes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import SAWITCSVDataset


class HerdingReplayBuffer:
    """Replay buffer that keeps samples closest to class centroids.

    This follows the core idea of the earlier program: compute class-wise feature
    centroids, store them as frozen prototypes, then choose exemplars nearest to
    each centroid.
    """

    def __init__(
        self,
        memory_limit: int,
        image_root: str | Path,
        device: str | torch.device,
        image_col: str = "id",
        label_col: str = "label",
        image_size: int = 112,
        min_per_class: int = 0,
        allow_missing_images: bool = False,
    ):
        self.memory_limit = int(memory_limit)
        self.image_root = Path(image_root)
        self.device = torch.device(device)
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = int(image_size)
        self.min_per_class = int(min_per_class)
        self.allow_missing_images = bool(allow_missing_images)

        self.buffer_df = pd.DataFrame(columns=[self.image_col, self.label_col])
        self.prototypes: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.buffer_df)

    def _allocation_per_class(self, num_classes: int) -> int:
        if num_classes <= 0 or self.memory_limit <= 0:
            return 0
        base = max(1, self.memory_limit // num_classes)
        if self.min_per_class > 0 and self.min_per_class * num_classes <= self.memory_limit:
            base = max(base, self.min_per_class)
        return base

    def update(self, model, dataloader: DataLoader, label_map: Dict[str, int]) -> None:
        if self.memory_limit <= 0:
            self.buffer_df = pd.DataFrame(columns=[self.image_col, self.label_col])
            self.prototypes = {}
            return

        model.eval()
        reverse_map = {v: k for k, v in label_map.items()}
        features_by_label: Dict[int, list] = {}
        paths_by_label: Dict[int, list] = {}

        with torch.no_grad():
            for x, y, paths in dataloader:
                x = x.to(self.device)
                features = model(x)
                if isinstance(features, tuple):
                    features = features[0]
                for i in range(len(y)):
                    label = int(y[i].detach().cpu().item())
                    features_by_label.setdefault(label, []).append(features[i].detach().cpu().numpy())
                    paths_by_label.setdefault(label, []).append(str(paths[i]))

        if not features_by_label:
            return

        rows = []
        k_per_class = self._allocation_per_class(len(features_by_label))
        self.prototypes = {}

        for label, features in features_by_label.items():
            feats = np.asarray(features, dtype=np.float32)
            centroid = feats.mean(axis=0)
            self.prototypes[label] = torch.tensor(centroid, dtype=torch.float32, device=self.device)

            distances = np.linalg.norm(feats - centroid, axis=1)
            selected = np.argsort(distances)[: min(k_per_class, len(distances))]
            for idx in selected:
                rows.append({self.image_col: paths_by_label[label][idx], self.label_col: reverse_map[label]})

        # Safety trim in case rounding/minimum allocation exceeds memory_limit.
        if len(rows) > self.memory_limit:
            rows = rows[: self.memory_limit]
        self.buffer_df = pd.DataFrame(rows, columns=[self.image_col, self.label_col])

    def as_dataset(self, label_map: Dict[str, int]):
        return SAWITCSVDataset(
            csv_file_or_df=self.buffer_df,
            image_root=self.image_root,
            label_map=label_map,
            image_col=self.image_col,
            label_col=self.label_col,
            image_size=self.image_size,
            allow_missing_images=self.allow_missing_images,
        )

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_df.to_csv(path, index=False)
