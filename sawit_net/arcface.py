"""ArcFace classification head used by SAWIT-Net."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ArcFaceLayer(nn.Module):
    """Additive angular margin head.

    During training, labels are required so the angular margin can be applied only
    to the ground-truth class. During inference, use ``cosine_logits``.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        if out_features < 1:
            raise ValueError("out_features must be >= 1")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.m = float(m)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None:
            raise ValueError("ArcFaceLayer.forward requires labels during training.")

        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Inference logits using normalized cosine similarity."""
        return F.linear(F.normalize(embeddings), F.normalize(self.weight)) * self.s

    def expand(self, new_out_features: int) -> "ArcFaceLayer":
        """Return a new head with more classes while preserving old weights."""
        new_out_features = int(new_out_features)
        if new_out_features <= self.out_features:
            return self

        new_layer = ArcFaceLayer(
            in_features=self.in_features,
            out_features=new_out_features,
            s=self.s,
            m=self.m,
        ).to(self.weight.device)
        with torch.no_grad():
            new_layer.weight[: self.out_features].copy_(self.weight.data)
        return new_layer
