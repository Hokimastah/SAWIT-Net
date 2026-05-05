"""Evaluation metrics for SAWIT-Net."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report


@torch.no_grad()
def predict(model, dataloader, device, use_prototypes: bool = False, prototype_temperature: float = 1.0):
    model.eval()
    y_true, y_pred = [], []
    for x, y, _ in dataloader:
        x = x.to(device)
        if use_prototypes:
            logits = model.predict_prototype_logits(x, temperature=prototype_temperature)
        else:
            logits = model.predict_logits(x)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def _metric_result(y_true, y_pred, report: bool = False) -> Dict[str, object]:
    # Compute macro metrics only on labels present in y_true. This fixes the
    # misleading low macro-F1 on base-only subsets after class expansion.
    labels = sorted(np.unique(y_true).tolist())
    result: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }
    if report:
        result["classification_report"] = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return result


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    report: bool = False,
    use_prototypes: bool = False,
    prototype_temperature: float = 1.0,
) -> Dict[str, object]:
    y_true, y_pred = predict(
        model,
        dataloader,
        device,
        use_prototypes=use_prototypes,
        prototype_temperature=prototype_temperature,
    )
    return _metric_result(y_true, y_pred, report=report)


def forgetting_score(base_accuracy_before: float, base_accuracy_after: float) -> float:
    """Simple forgetting score: base-session accuracy drop after incremental training."""
    return float(base_accuracy_before - base_accuracy_after)
