# SAWIT-Net

![alt text](<skip it/SAWIT-Net ikon.png>)

**SAWIT-Net** stands for **Self-Adaptive Weighted Incremental Transfer Network**.

SAWIT-Net is a PyTorch-based continual image classification toolkit designed to reduce **catastrophic forgetting** when a model learns new classes or new data sessions over time. It combines **ArcFace-based metric learning**, **herding replay**, **knowledge distillation**, **multi-scale feature preservation**, **contrastive distillation**, and **prototype preservation** in a single reusable training package.

This repository is designed as a standard Python tool/library, so it can be installed directly from GitHub, imported in Python scripts or notebooks, and executed from the command line.

---

## Core Idea

In standard fine-tuning, a model trained on old classes often performs well on new classes but forgets the old ones. This problem is known as **catastrophic forgetting**.

SAWIT-Net addresses this problem by combining several continual learning mechanisms:

1. **ArcFace Classification Head**  
   Learns discriminative embeddings using angular-margin classification. This is useful for identity-like, fine-grained, or visually similar image classification tasks.

2. **Herding Replay Buffer**  
   Stores representative samples from old classes by computing class centroids and selecting samples closest to each centroid.

3. **Embedding Knowledge Distillation**  
   Encourages the new model to preserve the old model's embedding representation.

4. **Multi-scale Feature Distillation**  
   Preserves intermediate feature maps from several backbone stages, not only the final embedding.

5. **Contrastive Knowledge Distillation**  
   Maintains similarity relationships between old and new feature representations.

6. **Prototype Preservation**  
   Keeps old-class embeddings close to frozen class prototypes obtained from the replay buffer.

The full SAWIT-Net objective is:

```text
L_total = L_cls
        + lambda_kd    * L_embedding_kd
        + lambda_ms    * L_multiscale_kd
        + lambda_ckd   * L_contrastive_kd
        + lambda_proto * L_prototype
```

---

## Architecture Overview

```text
Input Image
   │
   ▼
ResNet Backbone
   ├── layer2 feature map ─┐
   ├── layer3 feature map ─┼── Multi-scale Distillation
   └── layer4 feature map ─┘
   │
   ▼
Global Average Pooling
   │
   ▼
Embedding Layer
   │
   ├── ArcFace Classification Loss
   ├── Embedding Distillation
   ├── Contrastive Knowledge Distillation
   └── Prototype Preservation
```

Default configuration:

| Component | Default |
|---|---|
| Backbone | ResNet-50 |
| Input size | 112 × 112 |
| Embedding size | 512 |
| Classification head | ArcFace |
| Replay strategy | Herding replay |
| Main goal | Reducing catastrophic forgetting |

---

## Repository Structure

```text
SAWIT-Net/
├── sawit_net/
│   ├── __init__.py
│   ├── arcface.py
│   ├── buffer.py
│   ├── config.py
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── trainer.py
│   ├── cli.py
│   └── utils.py
├── examples/
│   ├── train_csv.py
│   ├── train_folder.py
│   └── colab_medmnist_demo.py
├── configs/
│   └── default.yaml
├── tests/
│   └── smoke_test.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

SAWIT-Net can be installed directly from GitHub.

### Google Colab

Run this command:

```python
!pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

Then test the import:

```python
from sawit_net import SAWITConfig, SAWITTrainer

print("SAWIT-Net imported successfully")
```

### Local Python / VS Code / Jupyter Notebook

Install directly from GitHub:

```bash
pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

For Jupyter Notebook, use the same Python kernel:

```python
import sys
!{sys.executable} -m pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

Then test:

```python
from sawit_net import SAWITConfig, SAWITTrainer

print("SAWIT-Net imported successfully")
```

---

## Upgrade Installation

If SAWIT-Net has already been installed and you want to reinstall the latest version from GitHub, use:

```bash
pip install --upgrade --force-reinstall git+https://github.com/Hokimastah/SAWIT-Net.git
```

For Google Colab:

```python
!pip install --upgrade --force-reinstall git+https://github.com/Hokimastah/SAWIT-Net.git
```

---

## Fixing `ModuleNotFoundError: No module named 'sawit_net'`

This error usually means SAWIT-Net has not been installed in the active Python environment.

### Google Colab

Run:

```python
!pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

Then import again:

```python
from sawit_net import SAWITConfig, SAWITTrainer
```

If the error still appears, restart the runtime and run the import again.

### Jupyter Notebook

Use the same Python executable as the active notebook kernel:

```python
import sys
!{sys.executable} -m pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

Then:

```python
from sawit_net import SAWITConfig, SAWITTrainer
```

Check the active Python environment:

```python
import sys
print(sys.executable)
```

---

## Supported Dataset Formats

SAWIT-Net supports two dataset formats:

1. CSV dataset
2. Folder dataset

---

## 1. CSV Dataset Format

Your CSV file must contain at least two columns:

```csv
id,label
person_001/img_001.jpg,person_001
person_001/img_002.jpg,person_001
person_002/img_001.jpg,person_002
```

Default columns:

| Column | Meaning |
|---|---|
| `id` | Relative or absolute image path |
| `label` | Class name |

Example folder structure:

```text
data/
├── images/
│   ├── person_001/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── person_002/
│       └── img_001.jpg
├── base.csv
└── incremental.csv
```

Example `base.csv`:

```csv
id,label
person_001/img_001.jpg,person_001
person_001/img_002.jpg,person_001
person_002/img_001.jpg,person_002
```

Example `incremental.csv`:

```csv
id,label
person_003/img_001.jpg,person_003
person_003/img_002.jpg,person_003
person_004/img_001.jpg,person_004
```

---

## 2. Folder Dataset Format

SAWIT-Net also supports folder datasets similar to `torchvision.datasets.ImageFolder`.

Example:

```text
data/
├── base/
│   ├── class_a/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── class_b/
│       └── 001.jpg
└── incremental/
    ├── class_c/
    │   └── 001.jpg
    └── class_d/
        └── 001.jpg
```

Each folder name becomes the class label.

---

## Quick Start: Python API

### CSV Training

```python
from sawit_net import SAWITConfig, SAWITTrainer

cfg = SAWITConfig(
    dataset_type="csv",
    image_root="./data/images",
    image_col="id",
    label_col="label",
    backbone="resnet50",
    head="arcface",
    image_size=112,
    emb_size=512,
    epochs=5,
    batch_size=32,
    lr=1e-4,
    memory_limit=500,
)

trainer = SAWITTrainer(cfg)

results = trainer.fit_two_stage(
    base_source="./data/base.csv",
    inc_source="./data/incremental.csv",
    mode="full",
)

print(results)

trainer.save("./checkpoints/sawit_net.pth")
```

### Folder Training

```python
from sawit_net import SAWITConfig, SAWITTrainer

cfg = SAWITConfig(
    dataset_type="folder",
    backbone="resnet50",
    head="arcface",
    image_size=112,
    emb_size=512,
    epochs=5,
    batch_size=32,
    lr=1e-4,
    memory_limit=500,
)

trainer = SAWITTrainer(cfg)

results = trainer.fit_two_stage(
    base_source="./data/base",
    inc_source="./data/incremental",
    mode="full",
)

print(results)

trainer.save("./checkpoints/sawit_net_folder.pth")
```

---

## Command Line Usage

After installation, SAWIT-Net provides a CLI command:

```bash
sawit-train
```

### CSV Dataset

```bash
sawit-train \
  --dataset-type csv \
  --image-root ./data/images \
  --base ./data/base.csv \
  --incremental ./data/incremental.csv \
  --mode full \
  --epochs 5 \
  --batch-size 32 \
  --memory-limit 500 \
  --output checkpoints/sawit_net.pth
```

### Folder Dataset

```bash
sawit-train \
  --dataset-type folder \
  --base ./data/base \
  --incremental ./data/incremental \
  --mode full \
  --epochs 5 \
  --batch-size 32 \
  --memory-limit 500 \
  --output checkpoints/sawit_net_folder.pth
```

The CLI saves the trained model to the selected output path and saves metrics to:

```text
checkpoints/metrics.json
```

---

## Training Modes

SAWIT-Net includes four experimental modes.

| Mode | Replay | KD | Multi-scale KD | Contrastive KD | Prototype Loss | Description |
|---|---:|---:|---:|---:|---:|---|
| `finetune` | No | No | No | No | No | Baseline. Trains only on new data. Usually causes catastrophic forgetting. |
| `replay_only` | Yes | No | No | No | No | Uses old representative samples without distillation. |
| `kd_only` | No | Yes | Yes | Yes | No | Uses teacher-student distillation without replay data. |
| `full` | Yes | Yes | Yes | Yes | Yes | Complete SAWIT-Net method. |

Example comparison:

```python
from sawit_net import SAWITConfig, SAWITTrainer

cfg = SAWITConfig(
    dataset_type="csv",
    image_root="./data/images",
    epochs=5,
    batch_size=32,
    memory_limit=500,
)

for mode in ["finetune", "replay_only", "kd_only", "full"]:
    trainer = SAWITTrainer(cfg)

    results = trainer.fit_two_stage(
        base_source="./data/base.csv",
        inc_source="./data/incremental.csv",
        mode=mode,
    )

    print(mode, results)
```

---

## Output Metrics

`fit_two_stage()` returns a dictionary like this:

```python
{
    "mode": "full",
    "num_classes": 10,
    "buffer_size": 500,
    "base_before": {
        "accuracy": 0.93,
        "macro_f1": 0.92,
        "macro_recall": 0.91
    },
    "base_after": {
        "accuracy": 0.88,
        "macro_f1": 0.87,
        "macro_recall": 0.86
    },
    "incremental_after": {
        "accuracy": 0.91,
        "macro_f1": 0.90,
        "macro_recall": 0.89
    },
    "all_after": {
        "accuracy": 0.89,
        "macro_f1": 0.88,
        "macro_recall": 0.87
    },
    "forgetting_score": 0.05
}
```

Metric explanation:

| Metric | Meaning |
|---|---|
| `accuracy` | Correct predictions divided by all predictions |
| `macro_f1` | Average F1 score across classes |
| `macro_recall` | Average recall across classes |
| `forgetting_score` | `base_before.accuracy - base_after.accuracy` |

A lower forgetting score is better.

---

## Configuration File

SAWIT-Net can use YAML configuration from `configs/default.yaml`.

Example:

```yaml
image_root: ./data/images
dataset_type: csv
image_col: id
label_col: label
image_size: 112

backbone: resnet50
pretrained: true
emb_size: 512
head: arcface
arc_s: 30.0
arc_m: 0.5

epochs: 5
batch_size: 32
lr: 0.0001
memory_limit: 500

kd_weight: 0.30
ms_weight: 0.30
ckd_weight: 0.20
proto_weight: 0.15
ckd_temperature: 2.0
```

Run with config:

```bash
sawit-train \
  --config configs/default.yaml \
  --base ./data/base.csv \
  --incremental ./data/incremental.csv \
  --mode full
```

Command-line arguments override the YAML values.

---

## Important Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| `backbone` | `resnet50` | CNN backbone. Supports `resnet18`, `resnet34`, `resnet50`, and `resnet101`. |
| `emb_size` | `512` | Final embedding dimension. |
| `head` | `arcface` | Classification head. Supports `arcface` or `linear`. |
| `arc_s` | `30.0` | ArcFace scale. |
| `arc_m` | `0.5` | ArcFace angular margin. |
| `memory_limit` | `500` | Maximum number of replay samples. |
| `kd_weight` | `0.30` | Embedding distillation weight. |
| `ms_weight` | `0.30` | Multi-scale distillation weight. |
| `ckd_weight` | `0.20` | Contrastive KD weight. |
| `proto_weight` | `0.15` | Prototype preservation weight. |
| `ckd_temperature` | `2.0` | Contrastive KD temperature. |

---

## Saving and Loading

### Save

```python
trainer.save("checkpoints/sawit_net.pth")
```

The checkpoint stores:

- model weights
- configuration
- label map
- replay buffer metadata
- class prototypes

### Load

```python
from sawit_net import SAWITTrainer

trainer = SAWITTrainer.load("checkpoints/sawit_net.pth")
```

---

## Google Colab Example

Install SAWIT-Net:

```python
!pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

Import:

```python
from sawit_net import SAWITConfig, SAWITTrainer
```

Prepare configuration:

```python
cfg = SAWITConfig(
    dataset_type="csv",
    image_root="./data/images",
    image_col="id",
    label_col="label",
    backbone="resnet50",
    head="arcface",
    image_size=112,
    emb_size=512,
    epochs=5,
    batch_size=32,
    lr=1e-4,
    memory_limit=500,
)
```

Train:

```python
trainer = SAWITTrainer(cfg)

results = trainer.fit_two_stage(
    base_source="./data/base.csv",
    inc_source="./data/incremental.csv",
    mode="full",
)

print(results)
```

Save:

```python
trainer.save("./sawit_net.pth")
```

---

## MedMNIST Example

Install dependencies:

```python
!pip install medmnist
!pip install git+https://github.com/Hokimastah/SAWIT-Net.git
```

Then import:

```python
from sawit_net import SAWITConfig, SAWITTrainer
```

You can use MedMNIST by preparing the dataset into either:

1. CSV format
2. Folder format

For CSV format, create two CSV files:

```text
base.csv
incremental.csv
```

Each CSV must contain:

```csv
id,label
image_001.png,class_0
image_002.png,class_1
```

Then train using the normal CSV API.

---

## Recommended Experiment Protocol

For a fair comparison, run all four modes on the same base and incremental split:

```bash
sawit-train --dataset-type csv --image-root ./data/images --base ./data/base.csv --incremental ./data/incremental.csv --mode finetune
sawit-train --dataset-type csv --image-root ./data/images --base ./data/base.csv --incremental ./data/incremental.csv --mode replay_only
sawit-train --dataset-type csv --image-root ./data/images --base ./data/base.csv --incremental ./data/incremental.csv --mode kd_only
sawit-train --dataset-type csv --image-root ./data/images --base ./data/base.csv --incremental ./data/incremental.csv --mode full
```

Compare:

1. `base_after.accuracy`
2. `incremental_after.accuracy`
3. `all_after.accuracy`
4. `forgetting_score`

A strong continual learning method should maintain high old-class accuracy while still learning the new classes.

---

## Notes and Limitations

- SAWIT-Net is designed for supervised image classification.
- The default ArcFace setup is especially suitable for identity-like recognition, fine-grained classes, or datasets where discriminative embeddings matter.
- For general medical image classification or ordinary object classification, the `linear` head can also be tested.
- Very small replay buffers may reduce old-class retention.
- `kd_only` may not be enough if the new data distribution is very different from the old data distribution.
- Training with ResNet-50 is GPU-recommended.
- For small datasets, reduce the batch size and number of epochs first before increasing model complexity.

---

## Minimal Import Example

```python
from sawit_net import SAWITConfig, SAWITTrainer

cfg = SAWITConfig()
trainer = SAWITTrainer(cfg)
```

## Citation

If you use SAWIT-Net in a project, thesis, or publication, please cite this repository:

```text
SAWIT-Net: Self-Adaptive Weighted Incremental Transfer Network.
GitHub Repository: https://github.com/Hokimastah/SAWIT-Net
```