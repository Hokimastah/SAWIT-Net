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

def prototype_relation_distillation_loss(
    new_feat: torch.Tensor,
    old_feat: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    PRD-style Prototype-Sample Relation Distillation.

    This loss preserves the relative similarity distribution between
    old class prototypes and current mini-batch samples.

    Unlike ordinary KD, this does not force the student feature to be
    identical to the teacher feature. Instead, it preserves how old
    prototypes relate to the current batch.

    Args:
        new_feat:
            Student/current model embeddings for the current batch.
        old_feat:
            Teacher/previous model embeddings for the same current batch.
        prototypes:
            Frozen old-class prototypes from previous sessions.
        temperature:
            Softmax temperature.

    Returns:
        KL divergence between old prototype-sample relation distribution
        and current prototype-sample relation distribution.
    """
    if len(prototypes) == 0:
        return torch.tensor(0.0, device=new_feat.device)

    proto_tensor = torch.stack(
        [p.detach().to(new_feat.device) for p in prototypes.values()]
    )

    proto_tensor = F.normalize(proto_tensor, dim=1)
    new_feat = F.normalize(new_feat, dim=1)
    old_feat = F.normalize(old_feat.detach(), dim=1)

    # Similarity of each old prototype to each sample in the mini-batch.
    # Shape: [num_old_prototypes, batch_size]
    current_logits = torch.matmul(proto_tensor, new_feat.T) / float(temperature)
    old_logits = torch.matmul(proto_tensor, old_feat.T) / float(temperature)

    # Distill the prototype-sample relation distribution.
    current_log_prob = F.log_softmax(current_logits, dim=1)
    old_prob = F.softmax(old_logits, dim=1).detach()

    return F.kl_div(current_log_prob, old_prob, reduction="batchmean")