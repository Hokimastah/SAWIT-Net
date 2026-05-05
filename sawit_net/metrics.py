from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_forgetting_score(acc_matrix: np.ndarray, current_task: int) -> float:
    if current_task == 0:
        return 0.0

    forgetting_values = []

    for old_task in range(current_task):
        history = acc_matrix[:current_task + 1, old_task]
        history = history[~np.isnan(history)]

        if len(history) == 0:
            continue

        best_acc = np.max(history)
        current_acc = acc_matrix[current_task, old_task]

        if not np.isnan(current_acc):
            forgetting_values.append(best_acc - current_acc)

    if len(forgetting_values) == 0:
        return 0.0

    return float(np.mean(forgetting_values))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    device,
    metric_labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    model.eval()

    y_true = []
    y_pred = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    if metric_labels is None:
        metric_labels = sorted(list(set(y_true)))

    acc = accuracy_score(y_true, y_pred)

    recall = recall_score(
        y_true,
        y_pred,
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
