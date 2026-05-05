"""Evaluation metrics for SAWIT-Net."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y, _ in dataloader:
        x = x.to(device)
        logits = model.predict_logits(x)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


@torch.no_grad()
def evaluate(model, dataloader, device, report: bool = False) -> Dict[str, object]:
    y_true, y_pred = predict(model, dataloader, device)
    result: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if report:
        result["classification_report"] = classification_report(y_true, y_pred, zero_division=0)
    return result


def forgetting_score(base_accuracy_before: float, base_accuracy_after: float) -> float:
    """Simple forgetting score: base-session accuracy drop after incremental training."""
    return float(base_accuracy_before - base_accuracy_after)
