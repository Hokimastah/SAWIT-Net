"""SAWIT-Net: Strategic Adaptive Weight Integration and Transfer Network."""

from .arcface import ArcFaceLayer
from .buffer import HerdingReplayBuffer
from .config import SAWITConfig
from .datasets import SAWITCSVDataset, SAWITFolderDataset
from .metrics import evaluate, forgetting_score, predict
from .trainer import SAWITTrainer

__version__ = "0.1.0"


def __getattr__(name):
    if name == "SAWITModel":
        from .models import SAWITModel

        return SAWITModel
    raise AttributeError(f"module 'sawit_net' has no attribute {name!r}")


__all__ = [
    "ArcFaceLayer",
    "HerdingReplayBuffer",
    "SAWITConfig",
    "SAWITCSVDataset",
    "SAWITFolderDataset",
    "SAWITModel",
    "SAWITTrainer",
    "evaluate",
    "forgetting_score",
    "predict",
]
