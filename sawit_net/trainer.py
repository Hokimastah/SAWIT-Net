"""Main trainer for SAWIT-Net."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim

from .buffer import HerdingReplayBuffer
from .config import SAWITConfig
from .datasets import build_dataset
from .losses import (
    contrastive_kd_loss,
    embedding_distillation_loss,
    multiscale_distillation_loss,
    prototype_preservation_loss,
    prototype_relation_distillation_loss,
)
from .metrics import evaluate, evaluate_with_prototypes, forgetting_score
from .utils import resolve_device, set_seed


class SAWITTrainer:
    """Trainer for two-stage and incremental continual learning.

    Main modes:
    - ``full``: replay + embedding KD + multi-scale KD + contrastive KD + prototype preservation.
    - ``replay_only``: replay + classification loss only.
    - ``kd_only``: new data + KD losses only, without replay buffer.
    - ``finetune``: new data + classification loss only.
    """

    VALID_MODES = {"full", "replay_only", "prd_only", "kd_only", "finetune"}
    
    def __init__(self, cfg: SAWITConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)

        self.label_map: Dict[str, int] = {}
        self.model = None
        self.buffer = HerdingReplayBuffer(
            memory_limit=cfg.memory_limit,
            image_root=cfg.image_root,
            device=self.device,
            image_col=cfg.image_col,
            label_col=cfg.label_col,
            image_size=cfg.image_size,
            min_per_class=cfg.min_per_class,
            allow_missing_images=cfg.allow_missing_images,
        )

    def _log(self, message: str) -> None:
        if self.cfg.verbose:
            print(message)

    def _make_loader(self, dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _make_optimizer(self):
        if self.model is None:
            raise RuntimeError("Model has not been created.")
        name = self.cfg.optimizer.lower()
        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.cfg.lr,
                momentum=0.9,
                weight_decay=self.cfg.weight_decay,
            )
        raise ValueError("optimizer must be 'adam' or 'sgd'.")

    def _create_model(self) -> None:
        if len(self.label_map) == 0:
            raise RuntimeError("Label map is empty. Build a dataset before creating the model.")
        from .models import SAWITModel

        self.model = SAWITModel(
            num_classes=len(self.label_map),
            backbone=self.cfg.backbone,
            emb_size=self.cfg.emb_size,
            pretrained=self.cfg.pretrained,
            head=self.cfg.head,
            arc_s=self.cfg.arc_s,
            arc_m=self.cfg.arc_m,
        ).to(self.device)

    def _expand_if_needed(self) -> None:
        if self.model is None:
            self._create_model()
        elif self.model.num_classes < len(self.label_map):
            self.model.expand_classes(len(self.label_map))
            self.model.to(self.device)

    def _classification_step(self, x: torch.Tensor, y: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model has not been created.")
        feat, logits = self.model(x, y)
        return feat, logits
    
    def _evaluate_loader(self, loader, mode: str, report: bool = False):
        """
        Use prototype-based evaluation for prd_only.
        Use normal classifier evaluation for other modes.
        """
        if mode == "prd_only":
            return evaluate_with_prototypes(
                self.model,
                loader,
                self.device,
                self.buffer.prototypes,
                report=report,
            )

        return evaluate(self.model, loader, self.device, report=report)
    
    @torch.no_grad()
    def _update_prototypes_only(self, loader, merge: bool = True) -> None:
        """
        Update class prototypes without storing replay images.

        This is used by prd_only mode. It keeps the method replay-free
        while still allowing old/new class prototypes to be maintained
        across sessions.
        """
        if self.model is None:
            raise RuntimeError("Model has not been created.")

        was_training = self.model.training
        self.model.eval()

        features_by_label = {}

        for x, y, _ in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            feat = self.model(x)

            if isinstance(feat, tuple):
                feat = feat[0]

            for i in range(len(y)):
                label = int(y[i].detach().cpu().item())
                features_by_label.setdefault(label, []).append(feat[i].detach())

        new_prototypes = {}

        for label, feats in features_by_label.items():
            feat_tensor = torch.stack(feats).to(self.device)
            new_prototypes[label] = feat_tensor.mean(dim=0).detach()

        if merge:
            self.buffer.prototypes.update(new_prototypes)
        else:
            self.buffer.prototypes = new_prototypes

        if was_training:
            self.model.train()

    def fit_base(self, source):
        """Train the base/offline session."""
        dataset = build_dataset(source, self.cfg, self.label_map)
        self._create_model()
        loader = self._make_loader(dataset, shuffle=True)
        optimizer = self._make_optimizer()
        ce = nn.CrossEntropyLoss()

        self._log(f"[SAWIT-Net] Base training: {len(dataset)} samples, {len(self.label_map)} classes")
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            total = 0.0
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, logits = self._classification_step(x, y)
                loss = ce(logits, y)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())
            self._log(f"  epoch {epoch}/{self.cfg.epochs} - loss={total / max(1, len(loader)):.4f}")

        # Store representative samples and prototypes from the base session.
        self.buffer.update(self.model, loader, self.label_map)
        return loader

    def fit_incremental(self, source, mode: str = "full"):
        """Train one incremental session."""
        mode = mode.lower()
        if mode == "kd_only":
            mode = "prd_only"

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        if self.model is None:
            raise RuntimeError("Call fit_base(...) before fit_incremental(...).")

        new_dataset = build_dataset(source, self.cfg, self.label_map)
        self._expand_if_needed()

        old_prototypes = {
            int(k): v.detach().clone().to(self.device)
            for k, v in self.buffer.prototypes.items()
        }

        use_replay = mode in {"full", "replay_only"} and len(self.buffer) > 0
        use_kd = mode == "full"
        use_prd = mode == "prd_only"
        use_proto = mode == "full"
        if use_prd:
            self.buffer.buffer_df = self.buffer.buffer_df.iloc[0:0].copy()

        if use_replay:
            replay_dataset = self.buffer.as_dataset(self.label_map)
            train_dataset = ConcatDataset([new_dataset, replay_dataset])
            self._log(f"[SAWIT-Net] Incremental training with replay: new={len(new_dataset)}, replay={len(replay_dataset)}")
        else:
            train_dataset = new_dataset
            self._log(f"[SAWIT-Net] Incremental training without replay: new={len(new_dataset)}")

        train_loader = self._make_loader(train_dataset, shuffle=True)
        new_loader = self._make_loader(new_dataset, shuffle=False)
        old_model = copy.deepcopy(self.model).eval()
        for p in old_model.parameters():
            p.requires_grad = False

        optimizer = self._make_optimizer()
        ce = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            running = {"cls": 0.0, "kd": 0.0, "ms": 0.0, "ckd": 0.0, "proto": 0.0, "total": 0.0}
            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                feat, logits = self._classification_step(x, y)
                loss_cls = ce(logits, y)
                loss = loss_cls

                loss_kd = torch.tensor(0.0, device=self.device)
                loss_ms = torch.tensor(0.0, device=self.device)
                loss_ckd = torch.tensor(0.0, device=self.device)
                loss_proto = torch.tensor(0.0, device=self.device)

                if use_kd:
                    new_feat, new_maps = self.model(x, return_features=True)
                    with torch.no_grad():
                        old_feat, old_maps = old_model(x, return_features=True)
                    loss_kd = embedding_distillation_loss(new_feat, old_feat)
                    loss_ms = multiscale_distillation_loss(new_maps, old_maps).to(self.device)
                    loss_ckd = contrastive_kd_loss(new_feat, old_feat, temperature=self.cfg.ckd_temperature)
                    loss = loss + self.cfg.kd_weight * loss_kd + self.cfg.ms_weight * loss_ms + self.cfg.ckd_weight * loss_ckd

                if use_prd and len(old_prototypes) > 0:
                    new_feat, _ = self.model(x, return_features=True)

                    with torch.no_grad():
                        old_feat, _ = old_model(x, return_features=True)

                    loss_kd = prototype_relation_distillation_loss(
                        new_feat=new_feat,
                        old_feat=old_feat,
                        prototypes=old_prototypes,
                        temperature=self.cfg.ckd_temperature,
                    )

                    loss = loss + self.cfg.kd_weight * loss_kd
                    
                if use_proto and len(self.buffer.prototypes) > 0:
                    loss_proto = prototype_preservation_loss(feat, y, self.buffer.prototypes)
                    loss = loss + self.cfg.proto_weight * loss_proto

                loss.backward()
                optimizer.step()

                running["cls"] += float(loss_cls.detach().cpu())
                running["kd"] += float(loss_kd.detach().cpu())
                running["ms"] += float(loss_ms.detach().cpu())
                running["ckd"] += float(loss_ckd.detach().cpu())
                running["proto"] += float(loss_proto.detach().cpu())
                running["total"] += float(loss.detach().cpu())

            denom = max(1, len(train_loader))
            self._log(
                f"  epoch {epoch}/{self.cfg.epochs} - "
                f"total={running['total']/denom:.4f}, cls={running['cls']/denom:.4f}, "
                f"kd={running['kd']/denom:.4f}, ms={running['ms']/denom:.4f}, "
                f"ckd={running['ckd']/denom:.4f}, proto={running['proto']/denom:.4f}"
            )

        # Update memory after incremental session.
        # prd_only must remain replay-free: update prototypes only, not image buffer.
        if use_prd:
            self._update_prototypes_only(new_loader, merge=True)
        else:
            self.buffer.update(self.model, train_loader, self.label_map)

        return new_loader

    def fit_two_stage(self, base_source, inc_source, mode: str = "full", report: bool = False) -> Dict[str, object]:
        """Convenience pipeline: base training -> incremental training -> evaluation."""
        requested_mode = mode.lower()
        actual_mode = "prd_only" if requested_mode == "kd_only" else requested_mode

        base_loader_before = self.fit_base(base_source)

        base_before = self._evaluate_loader(
            base_loader_before,
            mode=actual_mode,
            report=report,
        )

        self.fit_incremental(inc_source, mode=actual_mode)

        base_dataset_after = build_dataset(base_source, self.cfg, self.label_map)
        inc_dataset_after = build_dataset(inc_source, self.cfg, self.label_map)

        base_loader_after = self._make_loader(base_dataset_after, shuffle=False)
        inc_loader_after = self._make_loader(inc_dataset_after, shuffle=False)
        all_loader = self._make_loader(
            ConcatDataset([base_dataset_after, inc_dataset_after]),
            shuffle=False,
        )

        base_after = self._evaluate_loader(
            base_loader_after,
            mode=actual_mode,
            report=report,
        )

        inc_after = self._evaluate_loader(
            inc_loader_after,
            mode=actual_mode,
            report=report,
        )

        all_after = self._evaluate_loader(
            all_loader,
            mode=actual_mode,
            report=report,
        )

        return {
            "mode": actual_mode,
            "requested_mode": requested_mode,
            "num_classes": len(self.label_map),
            "buffer_size": len(self.buffer),
            "base_before": base_before,
            "base_after": base_after,
            "incremental_after": inc_after,
            "all_after": all_after,
            "forgetting_score": forgetting_score(
                base_before["accuracy"],
                base_after["accuracy"],
            ),
        }

    def evaluate_source(self, source, report: bool = False) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Model has not been created.")
        dataset = build_dataset(source, self.cfg, self.label_map)
        loader = self._make_loader(dataset, shuffle=False)
        return evaluate(self.model, loader, self.device, report=report)

    @torch.no_grad()
    def predict_loader(self, loader):
        if self.model is None:
            raise RuntimeError("Model has not been created.")
        from .metrics import predict

        return predict(self.model, loader, self.device)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.cfg.to_dict(),
            "label_map": self.label_map,
            "model_state": self.model.state_dict(),
            "buffer_df": self.buffer.buffer_df,
            "prototypes": {k: v.detach().cpu() for k, v in self.buffer.prototypes.items()},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "auto") -> "SAWITTrainer":
        device = resolve_device(map_location)
        payload = torch.load(path, map_location=device)
        cfg = SAWITConfig.from_dict(payload["config"])
        cfg.device = str(device)
        trainer = cls(cfg)
        trainer.label_map = dict(payload["label_map"])
        trainer._create_model()
        trainer.model.load_state_dict(payload["model_state"])
        trainer.model.to(trainer.device).eval()
        trainer.buffer.buffer_df = payload.get("buffer_df", trainer.buffer.buffer_df)
        trainer.buffer.prototypes = {int(k): v.to(trainer.device) for k, v in payload.get("prototypes", {}).items()}
        return trainer
