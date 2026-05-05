import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


def build_resnet_backbone(backbone_name: str, use_pretrained: bool):
    name = backbone_name.lower()

    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        return resnet18(weights=weights)

    if name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        return resnet34(weights=weights)

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        return resnet50(weights=weights)

    raise ValueError(
        f"Backbone '{backbone_name}' is not supported. "
        "Use 'resnet18', 'resnet34', or 'resnet50'."
    )


class ContinualResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 0,
        backbone_name: str = "resnet50",
        use_pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = build_resnet_backbone(backbone_name, use_pretrained)

        if in_channels != 3:
            old_conv = self.backbone.conv1

            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            if use_pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    else:
                        nn.init.kaiming_normal_(
                            new_conv.weight,
                            mode="fan_out",
                            nonlinearity="relu",
                        )

            self.backbone.conv1 = new_conv

        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.feature_dim = feature_dim
        self.num_classes = 0
        self.classifier = None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if num_classes > 0:
            self.expand_classifier(num_classes)

    def expand_classifier(self, new_num_classes: int):
        if new_num_classes <= self.num_classes:
            return

        device = next(self.parameters()).device
        new_classifier = nn.Linear(self.feature_dim, new_num_classes)

        if self.classifier is not None:
            with torch.no_grad():
                old_num_classes = self.num_classes
                new_classifier.weight[:old_num_classes].copy_(self.classifier.weight)
                new_classifier.bias[:old_num_classes].copy_(self.classifier.bias)

        self.classifier = new_classifier.to(device)
        self.num_classes = new_num_classes

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)
