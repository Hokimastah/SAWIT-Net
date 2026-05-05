from dataclasses import dataclass
from typing import List, Optional, Union


TaskClass = Union[int, str]


@dataclass
class CLConfig:
    # Strategy:
    # - "replay_kd"   : replay + knowledge distillation
    # - "replay_only" : replay only
    # - "kd_only"     : knowledge distillation only
    # - "finetune"    : no replay and no distillation
    strategy: str = "replay_kd"

    # Continual task split. You can use class indices or class names.
    tasks: Optional[List[List[TaskClass]]] = None
    base_task_classes: Optional[int] = None
    increment_classes: int = 2
    shuffle_class_order: bool = False

    # Model
    backbone: str = "resnet50"     # "resnet18", "resnet34", "resnet50"
    use_pretrained: bool = False
    freeze_backbone: bool = False

    # Training
    seed: int = 42
    batch_size: int = 32
    epochs_per_task: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"  # "adamw" or "sgd"

    # Replay memory
    memory_per_class: int = 100

    # Knowledge distillation
    kd_lambda: float = 1.0
    kd_temperature: float = 2.0

    # Dataloader
    num_workers: int = 0

    # Output
    output_dir: str = "outputs_cl"
    save_checkpoint: bool = True
    save_results: bool = True

    def use_replay(self) -> bool:
        return self.strategy.lower() in {"replay_kd", "replay_only"}

    def use_kd(self) -> bool:
        return self.strategy.lower() in {"replay_kd", "kd_only"}

    def validate(self):
        valid = {"replay_kd", "replay_only", "kd_only", "finetune"}
        if self.strategy.lower() not in valid:
            raise ValueError(
                f"strategy '{self.strategy}' is invalid. "
                f"Use one of: {sorted(valid)}"
            )

        valid_backbone = {"resnet18", "resnet34", "resnet50"}
        if self.backbone.lower() not in valid_backbone:
            raise ValueError(
                f"backbone '{self.backbone}' is invalid. "
                f"Use one of: {sorted(valid_backbone)}"
            )
