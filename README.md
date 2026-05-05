# SAWIT-Net

**SAWIT-Net** stands for **Self-Adaptive Weighted Incremental Transfer Network**.

![my icon](<skip it/SAWIT-Net ikon.png>)

SAWIT-Net is a lightweight continual learning library for image classification. It is designed to reduce catastrophic forgetting when a model learns new classes incrementally. The library supports replay memory, knowledge distillation, and their combination in one training pipeline.

The main objective of SAWIT-Net is to allow an image classification model to learn new classes over multiple training stages while preserving knowledge from previously learned classes.

---

## Key Features

- Continual learning for image classification
- Class-incremental learning workflow
- Replay memory for storing selected exemplars from previous classes
- Knowledge distillation from the previous model to the current model
- Four training strategies:
  - Replay + Knowledge Distillation
  - Replay only
  - Knowledge Distillation only
  - Fine-tuning baseline
- Dataset support:
  - ImageFolder
  - CSV-based dataset
  - Optional MedMNIST dataset
  - Custom PyTorch dataset
- Backbone support:
  - ResNet18
  - ResNet34
  - ResNet50
- Evaluation metrics:
  - Accuracy
  - Macro Recall
  - Macro F1-score
  - Forgetting Score

---

## Overview

In standard supervised learning, a model is usually trained on all classes at once. However, in real-world scenarios, new classes may arrive over time. Training only on new classes often causes the model to forget previously learned classes. This phenomenon is known as **catastrophic forgetting**.

SAWIT-Net addresses this problem using two main mechanisms:

1. **Replay Memory**  
   A small number of samples from previously learned classes are stored and replayed during later training stages.

2. **Knowledge Distillation**  
   The previous model is used as a teacher to guide the current model so that old knowledge is preserved.

General training flow:

```text
Task 0: Train on initial classes
Task 1: Train on new classes + optional replay + optional distillation
Task 2: Train on new classes + optional replay + optional distillation
...
```

---

## Architecture

The default architecture uses a ResNet backbone as the feature extractor and a dynamic classifier head that expands when new classes are introduced.

```text
Input Image
    ↓
ResNet Backbone
    ↓
Feature Embedding
    ↓
Expandable Classifier
    ↓
Class Prediction
```

When a new task is introduced, the classifier is expanded to include the new classes while preserving the classifier weights of previously learned classes.

---

## Available Strategies

SAWIT-Net provides four continual learning strategies.

| Strategy | Description |
|---|---|
| `replay_kd` | Uses replay memory and knowledge distillation |
| `replay_only` | Uses replay memory only |
| `kd_only` | Uses knowledge distillation only |
| `finetune` | Fine-tuning baseline without replay and without distillation |

Example:

```python
cfg = CLConfig(strategy="replay_kd")
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/sawit-net.git
cd sawit-net
```

Install the package in editable mode:

```bash
pip install -e .
```

Or install the dependencies manually:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main dependencies:

```text
torch
torchvision
scikit-learn
pandas
numpy
pillow
tqdm
```

For MedMNIST support, install:

```bash
pip install medmnist
```

---

## Project Structure

```text
sawit-net/
├── sawit_net/
│   ├── __init__.py
│   ├── config.py
│   ├── learner.py
│   ├── data.py
│   ├── memory.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   └── utils.py
│
├── examples/
│   ├── train_imagefolder.py
│   ├── train_csv.py
│   ├── train_medmnist.py
│   └── compare_strategies.py
│
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Import the Library

```python
from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_imagefolder_dataset
```

### 2. Load Dataset

```python
train_dataset, test_dataset, class_names, channels = load_imagefolder_dataset(
    train_dir="data/train",
    test_dir="data/test",
    img_size=224,
    input_mode="rgb",
)
```

### 3. Create Configuration

```python
cfg = CLConfig(
    strategy="replay_kd",
    backbone="resnet50",
    epochs_per_task=5,
    batch_size=32,
    memory_per_class=100,
    learning_rate=1e-4,
    output_dir="outputs_sawitnet",
)
```

### 4. Train the Model

```python
learner = ContinualLearner(
    config=cfg,
    class_names=class_names,
    in_channels=channels,
)

results = learner.fit(train_dataset, test_dataset)
print(results.tail())
```

---

## Dataset Format

SAWIT-Net supports several dataset formats.

---

## 1. ImageFolder Dataset

This is the easiest format to use.

The folder structure should be:

```text
data/
├── train/
│   ├── class_a/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── class_b/
│   │   ├── img003.jpg
│   │   └── img004.jpg
│   └── class_c/
│       ├── img005.jpg
│       └── img006.jpg
│
└── test/
    ├── class_a/
    │   ├── img101.jpg
    │   └── img102.jpg
    ├── class_b/
    │   ├── img103.jpg
    │   └── img104.jpg
    └── class_c/
        ├── img105.jpg
        └── img106.jpg
```

Example usage:

```python
from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_imagefolder_dataset

train_dataset, test_dataset, class_names, channels = load_imagefolder_dataset(
    train_dir="data/train",
    test_dir="data/test",
    img_size=224,
    input_mode="rgb",
)

cfg = CLConfig(
    strategy="replay_kd",
    backbone="resnet50",
    epochs_per_task=5,
    batch_size=32,
    memory_per_class=100,
)

learner = ContinualLearner(
    config=cfg,
    class_names=class_names,
    in_channels=channels,
)

results = learner.fit(train_dataset, test_dataset)
print(results)
```

---

## 2. CSV Dataset

The CSV file must contain at least two columns:

```csv
filepath,label
images/img001.jpg,class_a
images/img002.jpg,class_b
images/img003.jpg,class_c
```

Example usage:

```python
from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_csv_dataset

train_dataset, test_dataset, class_names, channels = load_csv_dataset(
    train_csv="data/train.csv",
    test_csv="data/test.csv",
    image_root="data",
    image_col="filepath",
    label_col="label",
    img_size=224,
    input_mode="rgb",
)

cfg = CLConfig(
    strategy="replay_only",
    backbone="resnet50",
    epochs_per_task=5,
    batch_size=32,
    memory_per_class=100,
)

learner = ContinualLearner(
    config=cfg,
    class_names=class_names,
    in_channels=channels,
)

results = learner.fit(train_dataset, test_dataset)
print(results)
```

---

## 3. MedMNIST Dataset

MedMNIST support is optional.

Install MedMNIST first:

```bash
pip install medmnist
```

Example usage:

```python
from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_medmnist_dataset

train_dataset, test_dataset, class_names, channels = load_medmnist_dataset(
    medmnist_name="organcmnist",
    img_size=128,
    input_mode="rgb",
)

cfg = CLConfig(
    strategy="kd_only",
    backbone="resnet50",
    epochs_per_task=5,
    batch_size=32,
    output_dir="outputs_medmnist_kd_only",
)

learner = ContinualLearner(
    config=cfg,
    class_names=class_names,
    in_channels=channels,
)

results = learner.fit(train_dataset, test_dataset)
print(results)
```

---

## 4. Custom PyTorch Dataset

SAWIT-Net can also work with a custom PyTorch dataset.

The dataset must return:

```python
image, label
```

where `label` is an integer class index.

The dataset should also provide one of the following attributes:

```python
dataset.targets
```

or:

```python
dataset.labels
```

Example:

```python
from torch.utils.data import Dataset

class MyImageDataset(Dataset):
    def __init__(self):
        self.samples = [...]
        self.targets = [...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = ...
        label = self.targets[index]
        return image, label
```

Then use:

```python
cfg = CLConfig(strategy="replay_kd")

learner = ContinualLearner(
    config=cfg,
    class_names=["class_a", "class_b", "class_c"],
    in_channels=3,
)

results = learner.fit(train_dataset, test_dataset)
```

---

## Continual Task Split

SAWIT-Net supports automatic and manual task splitting.

---

### Automatic Task Split

If `tasks=None`, SAWIT-Net automatically creates class-incremental tasks.

```python
cfg = CLConfig(
    tasks=None,
    base_task_classes=None,
    increment_classes=2,
)
```

By default:

```text
Task 0: approximately half of all classes
Task 1: next 2 classes
Task 2: next 2 classes
...
```

You can control the number of classes in the first task:

```python
cfg = CLConfig(
    base_task_classes=5,
    increment_classes=2,
)
```

This creates:

```text
Task 0: 5 classes
Task 1: 2 classes
Task 2: 2 classes
...
```

---

### Manual Task Split Using Class Indices

```python
cfg = CLConfig(
    tasks=[
        [0, 1, 2],
        [3, 4],
        [5, 6],
    ]
)
```

---

### Manual Task Split Using Class Names

```python
cfg = CLConfig(
    tasks=[
        ["normal", "pneumonia"],
        ["covid"],
        ["tuberculosis"],
    ]
)
```

The class names must match the folder names in ImageFolder or the labels in the CSV file.

---

## Training Strategies

---

### Replay + Knowledge Distillation

```python
cfg = CLConfig(
    strategy="replay_kd",
    memory_per_class=100,
    kd_lambda=1.0,
    kd_temperature=2.0,
)
```

This strategy uses both exemplar replay and teacher-student distillation.

---

### Replay Only

```python
cfg = CLConfig(
    strategy="replay_only",
    memory_per_class=100,
)
```

This strategy stores a fixed number of samples per class and reuses them when learning new tasks.

---

### Knowledge Distillation Only

```python
cfg = CLConfig(
    strategy="kd_only",
    kd_lambda=1.0,
    kd_temperature=2.0,
)
```

This strategy does not store old samples. The previous model is used as a teacher model during the next task.

---

### Fine-Tuning Baseline

```python
cfg = CLConfig(
    strategy="finetune",
)
```

This strategy trains only on the current task without replay memory and without knowledge distillation.

---

## Configuration Reference

The main configuration class is `CLConfig`.

```python
from sawit_net import CLConfig

cfg = CLConfig(
    strategy="replay_kd",
    tasks=None,
    base_task_classes=None,
    increment_classes=2,
    shuffle_class_order=False,

    backbone="resnet50",
    use_pretrained=False,
    freeze_backbone=False,

    seed=42,
    batch_size=32,
    epochs_per_task=5,
    learning_rate=1e-4,
    weight_decay=1e-4,
    optimizer_name="adamw",

    memory_per_class=100,

    kd_lambda=1.0,
    kd_temperature=2.0,

    num_workers=0,

    output_dir="outputs_cl",
    save_checkpoint=True,
    save_results=True,
)
```

### Important Parameters

| Parameter | Description |
|---|---|
| `strategy` | Continual learning strategy |
| `tasks` | Manual task split |
| `base_task_classes` | Number of classes in the first task |
| `increment_classes` | Number of new classes per incremental task |
| `backbone` | Backbone model |
| `use_pretrained` | Whether to use ImageNet-pretrained weights |
| `freeze_backbone` | Whether to freeze the backbone |
| `batch_size` | Training batch size |
| `epochs_per_task` | Number of epochs for each task |
| `learning_rate` | Learning rate |
| `memory_per_class` | Number of replay samples stored per class |
| `kd_lambda` | Weight for knowledge distillation loss |
| `kd_temperature` | Temperature for knowledge distillation |
| `output_dir` | Directory for checkpoints and results |

---

## Evaluation Metrics

SAWIT-Net reports the following metrics.

### Accuracy

Accuracy measures the proportion of correct predictions.

```text
Accuracy = Correct Predictions / Total Samples
```

### Macro Recall

Macro Recall computes recall independently for each class and then averages the result. This is useful for imbalanced datasets.

### Macro F1-score

Macro F1-score computes F1-score independently for each class and then averages the result. This is also useful for imbalanced datasets.

### Forgetting Score

Forgetting Score measures how much performance on old tasks decreases after learning new tasks.

For a task `j`:

```text
Forgetting(j) = Best Accuracy on Task j - Current Accuracy on Task j
```

The final forgetting score is the average forgetting across old tasks.

Lower forgetting score means better retention of old knowledge.

---

## Output Files

After training, SAWIT-Net saves outputs in the configured `output_dir`.

Example:

```text
outputs_cl/
├── checkpoint_after_task_0.pt
├── checkpoint_after_task_1.pt
├── checkpoint_after_task_2.pt
├── continual_learning_results.csv
└── accuracy_matrix.npy
```

### `continual_learning_results.csv`

This file contains:

| Column | Description |
|---|---|
| `strategy` | Training strategy |
| `after_training_task` | Task after which evaluation was performed |
| `eval_scope` | Evaluation scope |
| `eval_class_indices` | Evaluated class indices |
| `eval_class_names` | Evaluated class names |
| `accuracy` | Accuracy score |
| `recall_macro` | Macro recall |
| `f1_macro` | Macro F1-score |
| `forgetting_score` | Forgetting score |

### `accuracy_matrix.npy`

This file stores the per-task accuracy matrix.

```text
accuracy_matrix[t, j]
```

means:

```text
Accuracy on task j after finishing training task t
```

---

## Comparing Strategies

You can compare all available strategies using:

```python
import pandas as pd

from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_imagefolder_dataset

train_dataset, test_dataset, class_names, channels = load_imagefolder_dataset(
    train_dir="data/train",
    test_dir="data/test",
    img_size=224,
    input_mode="rgb",
)

all_results = []

for strategy in ["replay_kd", "replay_only", "kd_only", "finetune"]:
    cfg = CLConfig(
        strategy=strategy,
        backbone="resnet50",
        epochs_per_task=5,
        batch_size=32,
        memory_per_class=100,
        output_dir=f"outputs_{strategy}",
    )

    learner = ContinualLearner(
        config=cfg,
        class_names=class_names,
        in_channels=channels,
    )

    results = learner.fit(train_dataset, test_dataset)
    all_results.append(results)

summary = pd.concat(all_results, ignore_index=True)
summary.to_csv("strategy_comparison_results.csv", index=False)
print(summary)
```

---

## Example: Full Training Script

```python
from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_imagefolder_dataset


def main():
    train_dataset, test_dataset, class_names, channels = load_imagefolder_dataset(
        train_dir="data/train",
        test_dir="data/test",
        img_size=224,
        input_mode="rgb",
    )

    cfg = CLConfig(
        strategy="replay_kd",
        backbone="resnet50",
        use_pretrained=False,
        freeze_backbone=False,
        epochs_per_task=5,
        batch_size=32,
        learning_rate=1e-4,
        memory_per_class=100,
        kd_lambda=1.0,
        kd_temperature=2.0,
        output_dir="outputs_sawitnet",
    )

    learner = ContinualLearner(
        config=cfg,
        class_names=class_names,
        in_channels=channels,
    )

    results = learner.fit(train_dataset, test_dataset)
    print(results.tail())


if __name__ == "__main__":
    main()
```

---

## Recommended Experiment Setup

For research experiments, it is recommended to compare at least four strategies:

| Experiment | Strategy |
|---|---|
| Fine-tuning baseline | `finetune` |
| Replay only | `replay_only` |
| Knowledge distillation only | `kd_only` |
| Full SAWIT-Net | `replay_kd` |

The expected analysis should focus on:

- whether replay reduces forgetting,
- whether knowledge distillation preserves previous knowledge,
- whether the full SAWIT-Net strategy improves stability and final performance,
- how forgetting score changes after each incremental task.

---

## Troubleshooting

### 1. Class names in train and test are different

For ImageFolder datasets, the class folders in `train/` and `test/` must be identical.

Correct:

```text
train/class_a
train/class_b
test/class_a
test/class_b
```

Incorrect:

```text
train/class_a
train/class_b
test/class_a
test/class_c
```

---

### 2. CUDA out of memory

Reduce the batch size:

```python
cfg = CLConfig(batch_size=8)
```

or use a smaller backbone:

```python
cfg = CLConfig(backbone="resnet18")
```

---

### 3. Slow training on CPU

Use a smaller image size:

```python
img_size = 128
```

or use a smaller backbone:

```python
backbone = "resnet18"
```

---

### 4. Custom dataset does not work

Make sure the dataset returns:

```python
image, label
```

and has:

```python
dataset.targets
```

or:

```python
dataset.labels
```

---

## Limitations

SAWIT-Net is currently designed for image classification only.

It does not directly support:

- object detection,
- semantic segmentation,
- instance segmentation,
- tabular classification,
- audio classification,
- text classification.

For those tasks, the model head, dataset wrapper, and loss function need to be modified.

---

## Citation

If you use SAWIT-Net in your research, you may cite it as:

```bibtex
@misc{sawitnet2026,
  title        = {SAWIT-Net: Self-Adaptive Weighted Incremental Transfer Network for Continual Image Classification},
  author       = {Your Name},
  year         = {2026},
  howpublished = {GitHub Repository},
  url          = {https://github.com/your-username/sawit-net}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgement

SAWIT-Net is inspired by continual learning methods that use replay memory and knowledge distillation to reduce catastrophic forgetting in class-incremental learning scenarios.
