"""Tiny smoke test that checks lightweight imports."""

from sawit_net import SAWITConfig, SAWITTrainer


def test_imports():
    cfg = SAWITConfig(pretrained=False, backbone="resnet18", emb_size=64)
    trainer = SAWITTrainer(cfg)
    assert trainer.cfg.emb_size == 64


if __name__ == "__main__":
    test_imports()
    print("SAWIT-Net smoke test passed.")
