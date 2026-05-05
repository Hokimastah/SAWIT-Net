"""Command-line interface for SAWIT-Net."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SAWITConfig
from .trainer import SAWITTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAWIT-Net for two-stage continual image classification.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--dataset-type", type=str, default=None, choices=["csv", "folder"])
    parser.add_argument("--image-root", type=str, default=None, help="Root directory for CSV image paths.")
    parser.add_argument("--base", type=str, required=True, help="Base CSV path or base folder path.")
    parser.add_argument("--incremental", type=str, required=True, help="Incremental CSV path or incremental folder path.")
    parser.add_argument("--image-col", type=str, default=None, help="CSV image path column. Default: id")
    parser.add_argument("--label-col", type=str, default=None, help="CSV label column. Default: label")
    parser.add_argument("--mode",type=str,default="full",choices=["full", "replay_only", "prd_only", "kd_only", "finetune"])
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--head", type=str, default=None, choices=["arcface", "linear"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--memory-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="checkpoints/sawit_net.pth")
    parser.add_argument("--metrics-json", type=str, default="checkpoints/metrics.json")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Use zero tensors when images cannot be read.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    cfg = SAWITConfig.from_yaml(args.config) if args.config else SAWITConfig()

    # CLI overrides YAML/default values when provided.
    for key in [
        "dataset_type",
        "image_root",
        "image_col",
        "label_col",
        "backbone",
        "head",
        "epochs",
        "batch_size",
        "lr",
        "memory_limit",
        "device",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    if args.no_pretrained:
        cfg.pretrained = False
    if args.allow_missing_images:
        cfg.allow_missing_images = True

    trainer = SAWITTrainer(cfg)
    results = trainer.fit_two_stage(args.base, args.incremental, mode=args.mode, report=False)
    trainer.save(args.output)

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[SAWIT-Net] Training complete")
    print(json.dumps(results, indent=2))
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
