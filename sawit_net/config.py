"""Configuration objects for SAWIT-Net."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class SAWITConfig:
    """Main configuration for SAWIT-Net.

    The defaults follow the original experimental setup: ResNet-50, 512-dim
    embedding, ArcFace, 112x112 input, replay buffer, and multi-component KD.
    """

    # Data
    image_root: str = "."
    dataset_type: str = "csv"  # csv or folder
    image_col: str = "id"
    label_col: str = "label"
    image_size: int = 112
    allow_missing_images: bool = False

    # Model
    backbone: str = "resnet50"
    pretrained: bool = True
    emb_size: int = 512
    head: str = "arcface"  # arcface or linear
    arc_s: float = 30.0
    arc_m: float = 0.5

    # Training
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "adam"  # adam or sgd
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"

    # Continual learning
    memory_limit: int = 500
    min_per_class: int = 0
    kd_weight: float = 0.30
    ms_weight: float = 0.30
    ckd_weight: float = 0.20
    proto_weight: float = 0.15
    ckd_temperature: float = 2.0

    # Logging/checkpointing
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SAWITConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in values.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SAWITConfig":
        with open(path, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        return cls.from_dict(values)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
