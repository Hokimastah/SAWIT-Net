import os
import random
from typing import Dict, List, Sequence, Union

import numpy as np
import torch


TaskClass = Union[int, str]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_label(label) -> int:
    if torch.is_tensor(label):
        return int(label.view(-1)[0].item())
    return int(np.array(label).reshape(-1)[0])


def get_labels_from_dataset(dataset) -> np.ndarray:
    """
    Extract labels from a dataset.

    Supported:
    - torchvision.datasets.ImageFolder -> .targets
    - sawit_net.data.CSVDataset -> .targets
    - MedMNIST -> .labels
    """
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets).reshape(-1).astype(int)

    if hasattr(dataset, "labels"):
        return np.array(dataset.labels).reshape(-1).astype(int)

    raise AttributeError(
        "Dataset must have either .targets or .labels. "
        "For a custom dataset, add dataset.targets containing integer labels."
    )


def get_indices_by_classes(labels: np.ndarray, classes: List[int]) -> List[int]:
    class_set = set(int(c) for c in classes)
    return [idx for idx, y in enumerate(labels) if int(y) in class_set]


def resolve_task_class(item: TaskClass, class_names: Sequence[str]) -> int:
    if isinstance(item, int):
        if item < 0 or item >= len(class_names):
            raise ValueError(f"Class index {item} is outside 0..{len(class_names)-1}")
        return item

    if isinstance(item, str):
        if item not in class_names:
            raise ValueError(
                f"Class '{item}' was not found. Available classes: {list(class_names)}"
            )
        return int(class_names.index(item))

    raise TypeError(f"Unsupported class format: {item}")


def make_tasks(num_classes: int, class_names: Sequence[str], cfg) -> List[List[int]]:
    if cfg.tasks is not None:
        tasks = []
        for task in cfg.tasks:
            tasks.append([resolve_task_class(x, class_names) for x in task])
        return tasks

    class_order = list(range(num_classes))

    if cfg.shuffle_class_order:
        rng = random.Random(cfg.seed)
        rng.shuffle(class_order)

    if cfg.base_task_classes is None:
        base_size = max(1, num_classes // 2)
    else:
        base_size = int(cfg.base_task_classes)

    base_size = min(base_size, num_classes)
    inc = max(1, int(cfg.increment_classes))

    tasks = [class_order[:base_size]]
    remaining = class_order[base_size:]

    for i in range(0, len(remaining), inc):
        tasks.append(remaining[i:i + inc])

    return tasks


def build_class_mapping(tasks: List[List[int]]) -> Dict[int, int]:
    class_order = []
    for task in tasks:
        class_order.extend(task)

    if len(class_order) != len(set(class_order)):
        raise ValueError("A class appears more than once in tasks.")

    return {orig_cls: new_idx for new_idx, orig_cls in enumerate(class_order)}


def task_names(tasks: List[List[int]], class_names: Sequence[str]) -> List[List[str]]:
    return [[class_names[c] for c in task] for task in tasks]
