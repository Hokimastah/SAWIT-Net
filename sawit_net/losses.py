"""Continual-learning losses used by SAWIT-Net."""

from __future__ import annotations

from typing import Dict, List

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
        device = old_maps[0].device if old_maps else "cpu"
        return torch.tensor(0.0, device=device)

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


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss using same-label samples as positives.

    This is a single-view implementation. It works when each mini-batch contains
    at least two samples for some classes. If no positive pairs exist, it returns
    zero instead of NaN.
    """
    device = features.device
    labels = labels.view(-1)
    features = F.normalize(features, dim=1)

    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = features.size(0)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)

    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positives_per_row = positive_mask.sum(dim=1)
    valid = positives_per_row > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positives_per_row.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def prototype_tightness_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Prototype learning without negative contrasts.

    This follows the PRD paper's Lp idea: pull each class prototype toward its
    own class samples without pushing other class prototypes away. The sample
    features are stop-gradient here, so this term optimizes prototypes only.
    """
    if features.numel() == 0:
        return torch.tensor(0.0, device=features.device)
    selected_prototypes = prototypes[labels]
    return -(F.cosine_similarity(F.normalize(selected_prototypes, dim=1), F.normalize(features.detach(), dim=1))).mean()


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
    current_prototypes: torch.Tensor,
    old_prototypes: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Prototype-Sample Relation Distillation.

    For each old-class prototype, preserve its softmax distribution over the
    current mini-batch samples. Unlike the earlier simplified implementation,
    this compares old model + old prototypes against current model + current
    prototypes, so old prototypes can evolve with the representation.
    """
    if old_prototypes.numel() == 0:
        return torch.tensor(0.0, device=new_feat.device)

    n_old = old_prototypes.size(0)
    current_old_prototypes = current_prototypes[:n_old]

    old_proto = F.normalize(old_prototypes.detach().to(new_feat.device), dim=1)
    cur_proto = F.normalize(current_old_prototypes, dim=1)
    old_feat = F.normalize(old_feat.detach(), dim=1)
    new_feat = F.normalize(new_feat, dim=1)

    old_logits = torch.matmul(old_proto, old_feat.T) / float(temperature)
    cur_logits = torch.matmul(cur_proto, new_feat.T) / float(temperature)

    old_prob = F.softmax(old_logits, dim=1).detach()
    cur_log_prob = F.log_softmax(cur_logits, dim=1)
    return F.kl_div(cur_log_prob, old_prob, reduction="batchmean")
