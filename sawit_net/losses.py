"""Continual-learning losses used by SAWIT-Net."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def embedding_distillation_loss(new_feat: torch.Tensor, old_feat: torch.Tensor) -> torch.Tensor:
    """Cosine embedding distillation loss."""
    return (1.0 - F.cosine_similarity(new_feat, old_feat)).mean()


def multiscale_distillation_loss(new_maps: List[torch.Tensor], old_maps: List[torch.Tensor]) -> torch.Tensor:
    """Distill intermediate feature maps from multiple ResNet stages."""
    if len(new_maps) != len(old_maps):
        raise ValueError("new_maps and old_maps must have the same length.")
    if len(new_maps) == 0:
        return torch.tensor(0.0)

    losses = []
    for nf, of in zip(new_maps, old_maps):
        nf_flat = F.normalize(nf.flatten(1), dim=1)
        of_flat = F.normalize(of.flatten(1), dim=1)
        losses.append(F.mse_loss(nf_flat, of_flat))
    return torch.stack(losses).mean()


def contrastive_kd_loss(new_feat: torch.Tensor, old_feat: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Batch-wise contrastive distillation between student and teacher embeddings."""
    new_norm = F.normalize(new_feat, dim=1)
    old_norm = F.normalize(old_feat, dim=1)
    logits = torch.matmul(new_norm, old_norm.T) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def prototype_preservation_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Keep old-class embeddings close to frozen class prototypes."""
    proto_list = []
    feat_list = []
    for idx, label_tensor in enumerate(labels):
        label = int(label_tensor.detach().cpu().item())
        if label in prototypes:
            proto_list.append(prototypes[label].to(features.device))
            feat_list.append(features[idx])

    if not proto_list:
        return torch.tensor(0.0, device=features.device)

    proto_tensor = torch.stack(proto_list)
    feat_tensor = torch.stack(feat_list)
    return (1.0 - F.cosine_similarity(feat_tensor, proto_tensor)).mean()
