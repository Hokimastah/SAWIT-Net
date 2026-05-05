import random
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .data import MappedSubset


class ExemplarMemory:
    """
    Exemplar-based replay memory.
    Stores train indices from old classes.
    """
    def __init__(self, memory_per_class: int, seed: int = 42):
        self.memory_per_class = int(memory_per_class)
        self.memory: Dict[int, List[int]] = {}
        self.rng = random.Random(seed)

    def update(
        self,
        labels: np.ndarray,
        candidate_indices: List[int],
        classes: List[int],
    ):
        if self.memory_per_class <= 0:
            return

        for cls in classes:
            cls_indices = [
                idx for idx in candidate_indices
                if int(labels[idx]) == int(cls)
            ]

            self.rng.shuffle(cls_indices)
            selected = cls_indices[:self.memory_per_class]
            self.memory[int(cls)] = selected

    def get_all_indices(self) -> List[int]:
        all_indices = []
        for _, indices in self.memory.items():
            all_indices.extend(indices)
        return all_indices

    def __len__(self):
        return len(self.get_all_indices())

    def get_dataset(self, base_dataset, class_to_idx: Dict[int, int]) -> Optional[Dataset]:
        indices = self.get_all_indices()
        if len(indices) == 0:
            return None
        return MappedSubset(base_dataset, indices, class_to_idx)
