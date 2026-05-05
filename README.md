# SAWIT-Net

**SAWIT-Net** stands for **Strategic Adaptive Weight Integration and Transfer Network**.

SAWIT-Net is a PyTorch-based continual image classification toolkit designed to reduce **catastrophic forgetting** when a model learns new classes or new data sessions over time. It combines **ArcFace-based metric learning**, **herding replay**, **knowledge distillation**, **multi-scale feature preservation**, **contrastive distillation**, and **prototype preservation** in a single reusable training package.

This repository is written as a normal Python tool/library, so it can be installed, imported, reused in notebooks, or executed from the command line.

---

## Core Idea

In standard fine-tuning, a model trained on old classes often performs well on new classes but forgets the old ones. SAWIT-Net addresses this by combining five mechanisms:

1. **ArcFace classification head**  
   Learns discriminative embeddings using angular-margin classification. This is useful for identity-like or fine-grained image recognition tasks.

2. **Herding replay buffer**  
   Selects representative old samples by computing class centroids and keeping the samples closest to each centroid.

3. **Embedding knowledge distillation**  
   Forces the new model to preserve the old model's embedding representation.

4. **Multi-scale feature distillation**  
   Preserves intermediate feature maps from several ResNet stages, not only the final embedding.

5. **Prototype preservation**  
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
   ├── Contrastive KD
   └── Prototype Preservation
```

Default backbone: **ResNet-50**  
Default embedding size: **512**  
Default input size: **112 × 112**  
Default head: **ArcFace**

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

### 1. Clone or extract the project

```bash
git clone https://github.com/your-username/SAWIT-Net.git
cd SAWIT-Net
```

If you are using a ZIP file, extract it first:

```bash
unzip SAWIT-Net.zip
cd SAWIT-Net
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install SAWIT-Net as a local editable package

```bash
pip install -e .
```

### 4. Test the installation

```bash
python -c "import sawit_net; print(sawit_net.__version__)"
```

Expected output:

```text
0.1.0
```

---

## Fixing `ModuleNotFoundError: No module named 'sawit_net'`

This error means Python cannot find the package. Use one of these fixes.

### Local Python / VS Code / Jupyter

Run this from the root folder containing `pyproject.toml`:

```bash
pip install -e .
```

For Jupyter Notebook, use the same Python kernel:

```python
import sys
!{sys.executable} -m pip install -e .
```

### Google Colab

```python
from google.colab import files
uploaded = files.upload()
```

Upload `SAWIT-Net.zip`, then run:

```python
!unzip -q SAWIT-Net.zip -d /content
%cd /content/SAWIT-Net
!pip install -e .
```

Then test:

```python
from sawit_net import SAWITConfig, SAWITTrainer
print("SAWIT-Net imported successfully")
```

---

## Supported Dataset Formats

SAWIT-Net supports two data formats:

1. CSV dataset
2. ImageFolder-style dataset

---

## 1. CSV Dataset Format

Your CSV must contain at least two columns:

```csv
id,label
person_001/img_001.jpg,person_001
person_001/img_002.jpg,person_001
person_002/img_001.jpg,person_002
```

By default:

| Column | Meaning |
|---|---|
| `id` | Relative or absolute image path |
| `label` | Class name |

Example folder:

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

### CSV training

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

### Folder training

```python
from sawit_net import SAWITConfig, SAWITTrainer

cfg = SAWITConfig(
    dataset_type="folder",
    backbone="resnet50",
    head="arcface",
    epochs=5,
    batch_size=32,
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

After installation with `pip install -e .`, you can use the CLI command:

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

For folder datasets:

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

The CLI also saves metrics to:

```text
checkpoints/metrics.json
```

---

## Training Modes

SAWIT-Net includes four experimental modes.

| Mode | Replay | KD | Multi-scale KD | Contrastive KD | Prototype Loss | Description |
|---|---:|---:|---:|---:|---:|---|
| `finetune` | No | No | No | No | No | Baseline. Trains only on new data. Usually causes catastrophic forgetting. |
| `replay_only` | Yes | No | No | No | No | Uses old representative samples but no distillation. |
| `kd_only` | No | Yes | Yes | Yes | No | Uses teacher-student distillation without replay data. |
| `full` | Yes | Yes | Yes | Yes | Yes | Complete SAWIT-Net method. |

Example:

```python
for mode in ["finetune", "replay_only", "kd_only", "full"]:
    trainer = SAWITTrainer(cfg)
    results = trainer.fit_two_stage("base.csv", "incremental.csv", mode=mode)
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

You can use YAML configuration from `configs/default.yaml`.

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
| `backbone` | `resnet50` | CNN backbone. Supports `resnet18`, `resnet34`, `resnet50`, `resnet101`. |
| `emb_size` | `512` | Final embedding dimension. |
| `head` | `arcface` | Classification head: `arcface` or `linear`. |
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
- config
- label map
- replay buffer metadata
- class prototypes

### Load

```python
from sawit_net import SAWITTrainer

trainer = SAWITTrainer.load("checkpoints/sawit_net.pth")
```

---

## Google Colab Usage

Upload the ZIP file to Colab, then:

```python
!unzip -q SAWIT-Net.zip -d /content
%cd /content/SAWIT-Net
!pip install -e .
```

Check import:

```python
from sawit_net import SAWITConfig, SAWITTrainer
```

Run the CSV example:

```python
!python examples/train_csv.py
```

Run the MedMNIST demo:

```python
!pip install medmnist
!python examples/colab_medmnist_demo.py
```

---

## How to Upload to GitHub

From the `SAWIT-Net` folder:

```bash
git init
git add .
git commit -m "Initial release of SAWIT-Net"
git branch -M main
git remote add origin https://github.com/your-username/SAWIT-Net.git
git push -u origin main
```

If you already created the remote and get:

```text
error: remote origin already exists.
```

Use:

```bash
git remote set-url origin https://github.com/your-username/SAWIT-Net.git
git push -u origin main
```

---

## Recommended Experiment Protocol

For a fair comparison, run all four modes on the same split:

```bash
sawit-train --base base.csv --incremental incremental.csv --mode finetune
sawit-train --base base.csv --incremental incremental.csv --mode replay_only
sawit-train --base base.csv --incremental incremental.csv --mode kd_only
sawit-train --base base.csv --incremental incremental.csv --mode full
```

Then compare:

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

---

## Minimal Import Example

```python
from sawit_net import SAWITConfig, SAWITTrainer

cfg = SAWITConfig()
trainer = SAWITTrainer(cfg)
```

---

## License

This project is released under the MIT License.
