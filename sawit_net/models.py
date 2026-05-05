"""Backbone and SAWIT-Net model definitions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .arcface import ArcFaceLayer


_RESNET_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}

_RESNET_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
}


class SAWITModel(nn.Module):
    """ResNet embedding network with classifier and learnable prototypes.

    The normal classifier head is used by ``finetune``, ``replay_only``, and
    ``full``. The prototype bank is used by ``prd_only`` and also acts as an
    auxiliary stabilizer in ``full``.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        emb_size: int = 512,
        proj_size: int = 128,
        pretrained: bool = True,
        head: str = "linear",
        arc_s: float = 30.0,
        arc_m: float = 0.5,
    ):
        super().__init__()
        if backbone not in _RESNET_BUILDERS:
            raise ValueError(f"Unsupported backbone: {backbone}. Available: {sorted(_RESNET_BUILDERS)}")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        weights = _RESNET_WEIGHTS[backbone] if pretrained else None
        base = _RESNET_BUILDERS[backbone](weights=weights)

        self.backbone_name = backbone
        self.emb_size = int(emb_size)
        self.proj_size = int(proj_size)
        self.head_type = head.lower()

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.fc.in_features, emb_size)

        # Projection head for supervised contrastive representation learning.
        self.projector = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(emb_size, proj_size),
        )

        # Learnable class prototypes in embedding space. PRD uses these as the
        # classifier in replay-free mode.
        self.prototypes = nn.Parameter(torch.empty(num_classes, emb_size))
        nn.init.xavier_uniform_(self.prototypes)

        if self.head_type == "arcface":
            self.classifier = ArcFaceLayer(emb_size, num_classes, s=arc_s, m=arc_m)
        elif self.head_type == "linear":
            self.classifier = nn.Linear(emb_size, num_classes)
        else:
            raise ValueError("head must be 'arcface' or 'linear'.")

    @property
    def num_classes(self) -> int:
        if isinstance(self.classifier, ArcFaceLayer):
            return self.classifier.out_features
        return self.classifier.out_features

    def extract_features(self, x: torch.Tensor):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pooled = self.pool(f4).flatten(1)
        embedding = self.fc(pooled)
        return embedding, [f2, f3, f4]

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projector(embedding)

    def prototype_logits_from_embedding(self, embedding: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        feat = F.normalize(embedding, dim=1)
        proto = F.normalize(self.prototypes, dim=1)
        return torch.matmul(feat, proto.T) / float(temperature)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, return_features: bool = False):
        embedding, maps = self.extract_features(x)
        if return_features:
            return embedding, maps

        if isinstance(self.classifier, ArcFaceLayer):
            if labels is None:
                return embedding
            logits = self.classifier(embedding, labels)
        else:
            logits = self.classifier(embedding)

        if labels is not None:
            return embedding, logits
        return embedding

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.extract_features(x)
        if isinstance(self.classifier, ArcFaceLayer):
            return self.classifier.cosine_logits(embedding)
        return self.classifier(embedding)

    def predict_prototype_logits(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        embedding, _ = self.extract_features(x)
        return self.prototype_logits_from_embedding(embedding, temperature=temperature)

    def expand_classes(self, new_num_classes: int) -> None:
        """Expand classifier and prototype bank while preserving old weights."""
        new_num_classes = int(new_num_classes)
        if new_num_classes <= self.num_classes:
            return

        old_num_classes = self.num_classes

        # Expand normal classifier.
        if isinstance(self.classifier, ArcFaceLayer):
            self.classifier = self.classifier.expand(new_num_classes)
        else:
            old = self.classifier
            new = nn.Linear(old.in_features, new_num_classes).to(old.weight.device)
            with torch.no_grad():
                new.weight[: old.out_features].copy_(old.weight.data)
                new.bias[: old.out_features].copy_(old.bias.data)
            self.classifier = new

        # Expand learnable prototypes.
        old_proto = self.prototypes
        new_proto = nn.Parameter(torch.empty(new_num_classes, self.emb_size, device=old_proto.device))
        nn.init.xavier_uniform_(new_proto)
        with torch.no_grad():
            new_proto[:old_num_classes].copy_(old_proto.data)
        self.prototypes = new_proto
