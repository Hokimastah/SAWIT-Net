from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import knowledge_distillation_loss


def build_optimizer(model: nn.Module, cfg):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )

    if cfg.optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")


def train_one_task(
    model: nn.Module,
    teacher_model: Optional[nn.Module],
    train_loader,
    optimizer,
    device,
    cfg,
    task_id: int,
) -> Dict[str, float]:
    ce_loss_fn = nn.CrossEntropyLoss()

    model.train()

    if teacher_model is not None:
        teacher_model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_samples = 0

    use_kd = cfg.use_kd() and teacher_model is not None and cfg.kd_lambda > 0

    pbar = tqdm(train_loader, desc=f"Training Task {task_id}", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        student_logits = model(images)
        ce_loss = ce_loss_fn(student_logits, labels)

        kd_loss = torch.tensor(0.0, device=device)

        if use_kd:
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            kd_loss = knowledge_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=cfg.kd_temperature,
            )

        loss = ce_loss + cfg.kd_lambda * kd_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_kd += kd_loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{ce_loss.item():.4f}",
            "kd": f"{kd_loss.item():.4f}",
        })

    return {
        "loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce / max(1, total_samples),
        "kd_loss": total_kd / max(1, total_samples),
    }
