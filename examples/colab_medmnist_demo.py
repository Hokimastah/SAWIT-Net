"""Google Colab demo for MedMNIST-like arrays.

This example converts OrganCMNIST into a temporary image folder + CSV split,
then trains SAWIT-Net as a two-stage continual learning experiment.
"""

import os
from pathlib import Path

import pandas as pd
from PIL import Image

# In Colab, run first:
# !pip install medmnist
# !pip install -e .

from medmnist import OrganCMNIST
from sawit_net import SAWITConfig, SAWITTrainer


def export_medmnist_to_csv(root="./data/medmnist_sawit", size=128, max_train=None):
    root = Path(root)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    dataset = OrganCMNIST(split="train", download=True, size=size)

    rows = []
    for idx in range(len(dataset)):
        if max_train is not None and idx >= max_train:
            break
        img, label = dataset[idx]
        label_id = int(label[0]) if hasattr(label, "__len__") else int(label)
        rel_path = f"organ_{idx:06d}.png"
        Image.fromarray(img).convert("RGB").save(image_root / rel_path)
        rows.append({"id": rel_path, "label": f"class_{label_id}"})

    df = pd.DataFrame(rows)

    # Simple continual split by class: first half as base, remaining as incremental.
    labels = sorted(df["label"].unique())
    cut = max(1, len(labels) // 2)
    base_labels = set(labels[:cut])
    base_df = df[df["label"].isin(base_labels)].reset_index(drop=True)
    inc_df = df[~df["label"].isin(base_labels)].reset_index(drop=True)

    base_csv = root / "base.csv"
    inc_csv = root / "incremental.csv"
    base_df.to_csv(base_csv, index=False)
    inc_df.to_csv(inc_csv, index=False)
    return str(image_root), str(base_csv), str(inc_csv)


if __name__ == "__main__":
    image_root, base_csv, inc_csv = export_medmnist_to_csv(max_train=3000)

    cfg = SAWITConfig(
        dataset_type="csv",
        image_root=image_root,
        image_col="id",
        label_col="label",
        backbone="resnet50",
        head="arcface",
        image_size=112,
        emb_size=512,
        epochs=3,
        batch_size=32,
        lr=1e-4,
        memory_limit=500,
    )

    trainer = SAWITTrainer(cfg)
    results = trainer.fit_two_stage(base_csv, inc_csv, mode="full")
    print(results)
    trainer.save("./checkpoints/sawit_net_medmnist.pth")
