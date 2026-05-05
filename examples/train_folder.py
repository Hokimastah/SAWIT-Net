"""Example: training SAWIT-Net with ImageFolder-style datasets.

Folder format:
    data/base/class_a/001.jpg
    data/base/class_b/001.jpg
    data/incremental/class_c/001.jpg
"""

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
