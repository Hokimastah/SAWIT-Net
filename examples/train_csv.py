"""Example: training SAWIT-Net with CSV datasets.

Expected CSV format:
    id,label
    person_001/img1.jpg,person_001
    person_002/img1.jpg,person_002
"""

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
