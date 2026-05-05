import os
from typing import Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from .utils import parse_label


class CSVDataset(Dataset):
    """
    Image dataset based on a CSV file.

    The CSV file must contain at least two columns:
    - filepath: image path
    - label   : class label
    """
    def __init__(
        self,
        csv_path: str,
        image_root: str = "",
        image_col: str = "filepath",
        label_col: str = "label",
        label_to_idx: Dict[str, int] = None,
        transform=None,
        input_mode: str = "rgb",
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.input_mode = input_mode.lower()

        if image_col not in self.df.columns:
            raise ValueError(f"image_col '{image_col}' does not exist in {csv_path}")

        if label_col not in self.df.columns:
            raise ValueError(f"label_col '{label_col}' does not exist in {csv_path}")

        if label_to_idx is None:
            class_names = sorted(self.df[label_col].astype(str).unique().tolist())
            label_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.label_to_idx = label_to_idx
        self.classes = [None] * len(label_to_idx)
        for name, idx in label_to_idx.items():
            self.classes[idx] = name

        self.samples = []
        self.targets = []

        for _, row in self.df.iterrows():
            image_path = str(row[image_col])
            label_name = str(row[label_col])

            if label_name not in label_to_idx:
                raise ValueError(
                    f"Label '{label_name}' in {csv_path} does not exist in label_to_idx."
                )

            full_path = image_path
            if image_root:
                full_path = os.path.join(image_root, image_path)

            label_idx = label_to_idx[label_name]
            self.samples.append((full_path, label_idx))
            self.targets.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path)

        if self.input_mode == "grayscale":
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MappedSubset(Dataset):
    """
    Dataset subset with mapping from original class labels to incremental labels.
    """
    def __init__(self, base_dataset, indices: List[int], class_to_idx: Dict[int, int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        image, label = self.base_dataset[real_idx]

        original_label = parse_label(label)
        mapped_label = self.class_to_idx[original_label]

        return image, mapped_label


def build_image_transforms(img_size: int = 224, input_mode: str = "rgb"):
    channels = 1 if input_mode.lower() == "grayscale" else 3

    mean = [0.5] * channels
    std = [0.5] * channels

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform, channels


def load_imagefolder_dataset(
    train_dir: str,
    test_dir: str,
    img_size: int = 224,
    input_mode: str = "rgb",
):
    transform, channels = build_image_transforms(img_size, input_mode)

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Class folders in train and test must be identical. "
            f"Train classes: {train_dataset.classes}; Test classes: {test_dataset.classes}"
        )

    class_names = list(train_dataset.classes)
    return train_dataset, test_dataset, class_names, channels


def load_csv_dataset(
    train_csv: str,
    test_csv: str,
    image_root: str = "",
    image_col: str = "filepath",
    label_col: str = "label",
    img_size: int = 224,
    input_mode: str = "rgb",
):
    transform, channels = build_image_transforms(img_size, input_mode)

    train_df = pd.read_csv(train_csv)

    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' does not exist in {train_csv}")

    class_names = sorted(train_df[label_col].astype(str).unique().tolist())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_dataset = CSVDataset(
        csv_path=train_csv,
        image_root=image_root,
        image_col=image_col,
        label_col=label_col,
        label_to_idx=label_to_idx,
        transform=transform,
        input_mode=input_mode,
    )

    test_dataset = CSVDataset(
        csv_path=test_csv,
        image_root=image_root,
        image_col=image_col,
        label_col=label_col,
        label_to_idx=label_to_idx,
        transform=transform,
        input_mode=input_mode,
    )

    return train_dataset, test_dataset, class_names, channels


def load_medmnist_dataset(
    medmnist_name: str = "organcmnist",
    img_size: int = 128,
    input_mode: str = "rgb",
):
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise ImportError("medmnist is not installed. Run: pip install medmnist") from exc

    dataset_key = medmnist_name.lower()

    if dataset_key not in INFO:
        raise ValueError(f"MedMNIST dataset '{medmnist_name}' was not found.")

    info = INFO[dataset_key]

    if info["task"] != "multi-class":
        raise ValueError(
            f"This code is designed for multi-class classification. "
            f"Dataset {medmnist_name} has task: {info['task']}"
        )

    DataClass = getattr(medmnist, info["python_class"])
    transform, channels = build_image_transforms(img_size, input_mode)
    as_rgb = input_mode.lower() != "grayscale"

    train_dataset = DataClass(
        split="train",
        transform=transform,
        download=True,
        as_rgb=as_rgb,
        size=img_size,
    )

    test_dataset = DataClass(
        split="test",
        transform=transform,
        download=True,
        as_rgb=as_rgb,
        size=img_size,
    )

    labels_dict = info["label"]
    class_names = [str(labels_dict[str(i)]) for i in range(len(labels_dict))]

    return train_dataset, test_dataset, class_names, channels
