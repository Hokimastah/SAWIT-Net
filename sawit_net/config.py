"""Configuration objects for SAWIT-Net."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class SAWITConfig:
    """Main configuration for SAWIT-Net.

    SAWIT-Net supports both replay-based continual learning and a PRD-style
    replay-free mode. For ordinary image classification datasets such as
    MedMNIST, ``head='linear'`` is recommended. For identity-like recognition,
    ``head='arcface'`` can still be used.
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
    proj_size: int = 128
    head: str = "linear"  # linear or arcface
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

    # Replay memory
    memory_limit: int = 500
    min_per_class: int = 0

    # SAWIT/continual learning losses
    # Existing KD losses used mainly in full/replay-based mode.
    kd_weight: float = 0.10
    ms_weight: float = 0.00
    ckd_weight: float = 0.00

    # Prototype preservation for replay-based full mode.
    proto_weight: float = 0.30

    # PRD-style replay-free losses.
    supcon_weight: float = 1.00
    prototype_weight: float = 1.00
    prd_weight: float = 4.00
    supcon_temperature: float = 0.10
    prd_temperature: float = 0.20

    # Backward compatibility for old code using ckd_temperature.
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
