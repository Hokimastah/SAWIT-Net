"""Main trainer for SAWIT-Net."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

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
    prototype_tightness_loss,
    supervised_contrastive_loss,
)
from .metrics import evaluate, forgetting_score
from .utils import resolve_device, set_seed


class SAWITTrainer:
    """Trainer for two-stage and incremental continual learning.

    Modes:
    - ``finetune``: classifier CE on new data only.
    - ``replay_only``: replay + classifier CE.
    - ``prd_only``: replay-free PRD-style prototype contrastive learner.
    - ``full``: combined SAWIT method: replay + CE + KD + SupCon + PRD + prototype losses.

    ``kd_only`` is kept only as a backward-compatible alias to ``prd_only``.
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
            proj_size=self.cfg.proj_size,
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

    @torch.no_grad()
    def _sync_model_prototypes_to_buffer(self) -> None:
        """Mirror the model's learnable prototypes into the trainer buffer."""
        if self.model is None:
            return
        self.buffer.prototypes = {
            int(i): self.model.prototypes[i].detach().clone().to(self.device)
            for i in range(self.model.prototypes.size(0))
        }

    @torch.no_grad()
    def _sync_buffer_centroids_to_model_prototypes(self) -> None:
        """Initialize model prototypes from herding centroids when available."""
        if self.model is None or len(self.buffer.prototypes) == 0:
            return
        for label, proto in self.buffer.prototypes.items():
            if 0 <= int(label) < self.model.prototypes.size(0):
                self.model.prototypes[int(label)].copy_(proto.to(self.device))

    def _base_auxiliary_losses(self, feat: torch.Tensor, y: torch.Tensor):
        """SupCon + prototype learning used to prepare prototype space."""
        z = self.model.project(feat)
        loss_sc = supervised_contrastive_loss(z, y, temperature=self.cfg.supcon_temperature)
        loss_pt = prototype_tightness_loss(feat, y, self.model.prototypes)
        return loss_sc, loss_pt

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
            running = {"cls": 0.0, "sc": 0.0, "pt": 0.0, "total": 0.0}
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                feat, logits = self._classification_step(x, y)
                loss_cls = ce(logits, y)
                loss_sc, loss_pt = self._base_auxiliary_losses(feat, y)
                loss = loss_cls + self.cfg.supcon_weight * loss_sc + self.cfg.prototype_weight * loss_pt
                loss.backward()
                optimizer.step()
                running["cls"] += float(loss_cls.detach().cpu())
                running["sc"] += float(loss_sc.detach().cpu())
                running["pt"] += float(loss_pt.detach().cpu())
                running["total"] += float(loss.detach().cpu())
            denom = max(1, len(loader))
            self._log(
                f"  epoch {epoch}/{self.cfg.epochs} - total={running['total']/denom:.4f}, "
                f"cls={running['cls']/denom:.4f}, sc={running['sc']/denom:.4f}, pt={running['pt']/denom:.4f}"
            )

        # Store representative samples and centroids from the base session, then
        # initialize the learnable prototype bank from those centroids.
        self.buffer.update(self.model, loader, self.label_map)
        self._sync_buffer_centroids_to_model_prototypes()
        self._sync_model_prototypes_to_buffer()
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

        # Snapshot old model/prototypes before adding new labels/classes.
        old_num_classes = self.model.num_classes
        old_model = copy.deepcopy(self.model).eval()
        for p in old_model.parameters():
            p.requires_grad = False
        old_prototypes = self.model.prototypes[:old_num_classes].detach().clone().to(self.device)

        new_dataset = build_dataset(source, self.cfg, self.label_map)
        self._expand_if_needed()

        use_replay = mode in {"full", "replay_only"} and len(self.buffer) > 0
        use_prd = mode == "prd_only"
        use_full = mode == "full"

        if use_replay:
            replay_dataset = self.buffer.as_dataset(self.label_map)
            train_dataset = ConcatDataset([new_dataset, replay_dataset])
            self._log(f"[SAWIT-Net] Incremental training with replay: new={len(new_dataset)}, replay={len(replay_dataset)}")
        else:
            train_dataset = new_dataset
            self._log(f"[SAWIT-Net] Incremental training without replay: new={len(new_dataset)}")

        train_loader = self._make_loader(train_dataset, shuffle=True)
        new_loader = self._make_loader(new_dataset, shuffle=False)
        optimizer = self._make_optimizer()
        ce = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            running = {"cls": 0.0, "sc": 0.0, "pt": 0.0, "kd": 0.0, "ms": 0.0, "ckd": 0.0, "prd": 0.0, "proto": 0.0, "total": 0.0}
            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                feat, logits = self._classification_step(x, y)

                loss_cls = torch.tensor(0.0, device=self.device)
                loss_sc = torch.tensor(0.0, device=self.device)
                loss_pt = torch.tensor(0.0, device=self.device)
                loss_kd = torch.tensor(0.0, device=self.device)
                loss_ms = torch.tensor(0.0, device=self.device)
                loss_ckd = torch.tensor(0.0, device=self.device)
                loss_prd = torch.tensor(0.0, device=self.device)
                loss_proto = torch.tensor(0.0, device=self.device)

                if mode in {"finetune", "replay_only", "full"}:
                    loss_cls = ce(logits, y)

                # PRD-style representation/prototype learning is used by
                # prd_only and by full as the combined SAWIT method.
                if mode in {"prd_only", "full"}:
                    z = self.model.project(feat)
                    loss_sc = supervised_contrastive_loss(z, y, temperature=self.cfg.supcon_temperature)
                    loss_pt = prototype_tightness_loss(feat, y, self.model.prototypes)
                    with torch.no_grad():
                        old_feat, _ = old_model(x, return_features=True)
                    loss_prd = prototype_relation_distillation_loss(
                        new_feat=feat,
                        old_feat=old_feat,
                        current_prototypes=self.model.prototypes,
                        old_prototypes=old_prototypes,
                        temperature=self.cfg.prd_temperature,
                    )

                # Full remains a true combined method: replay + CE + PRD losses
                # + old-style KD/prototype preservation.
                if use_full:
                    new_feat, new_maps = self.model(x, return_features=True)
                    with torch.no_grad():
                        old_feat2, old_maps = old_model(x, return_features=True)
                    loss_kd = embedding_distillation_loss(new_feat, old_feat2)
                    loss_ms = multiscale_distillation_loss(new_maps, old_maps).to(self.device)
                    loss_ckd = contrastive_kd_loss(new_feat, old_feat2, temperature=self.cfg.ckd_temperature)
                    if len(self.buffer.prototypes) > 0:
                        loss_proto = prototype_preservation_loss(feat, y, self.buffer.prototypes)

                loss = (
                    loss_cls
                    + self.cfg.supcon_weight * loss_sc
                    + self.cfg.prototype_weight * loss_pt
                    + self.cfg.prd_weight * loss_prd
                    + self.cfg.kd_weight * loss_kd
                    + self.cfg.ms_weight * loss_ms
                    + self.cfg.ckd_weight * loss_ckd
                    + self.cfg.proto_weight * loss_proto
                )

                loss.backward()
                optimizer.step()

                for key, value in [
                    ("cls", loss_cls), ("sc", loss_sc), ("pt", loss_pt), ("kd", loss_kd),
                    ("ms", loss_ms), ("ckd", loss_ckd), ("prd", loss_prd), ("proto", loss_proto), ("total", loss),
                ]:
                    running[key] += float(value.detach().cpu())

            denom = max(1, len(train_loader))
            self._log(
                f"  epoch {epoch}/{self.cfg.epochs} - total={running['total']/denom:.4f}, "
                f"cls={running['cls']/denom:.4f}, sc={running['sc']/denom:.4f}, pt={running['pt']/denom:.4f}, "
                f"prd={running['prd']/denom:.4f}, kd={running['kd']/denom:.4f}, "
                f"ms={running['ms']/denom:.4f}, ckd={running['ckd']/denom:.4f}, proto={running['proto']/denom:.4f}"
            )

        # Update memory after incremental session.
        if use_prd:
            # replay-free: no image buffer, prototypes only.
            self.buffer.buffer_df = self.buffer.buffer_df.iloc[0:0].copy()
            self._sync_model_prototypes_to_buffer()
        else:
            self.buffer.update(self.model, train_loader, self.label_map)
            self._sync_buffer_centroids_to_model_prototypes()
            self._sync_model_prototypes_to_buffer()

        return new_loader

    def _eval_loader(self, loader, mode: str, report: bool = False):
        # PRD-only must be evaluated through nearest-prototype prediction.
        return evaluate(
            self.model,
            loader,
            self.device,
            report=report,
            use_prototypes=(mode == "prd_only"),
            prototype_temperature=self.cfg.prd_temperature,
        )

    def fit_two_stage(self, base_source, inc_source, mode: str = "full", report: bool = False) -> Dict[str, object]:
        """Convenience pipeline: base training -> incremental training -> evaluation."""
        requested_mode = mode.lower()
        mode = "prd_only" if requested_mode == "kd_only" else requested_mode

        base_loader_before = self.fit_base(base_source)
        base_before = self._eval_loader(base_loader_before, mode, report=report)

        self.fit_incremental(inc_source, mode=mode)

        base_dataset_after = build_dataset(base_source, self.cfg, self.label_map)
        inc_dataset_after = build_dataset(inc_source, self.cfg, self.label_map)
        base_loader_after = self._make_loader(base_dataset_after, shuffle=False)
        inc_loader_after = self._make_loader(inc_dataset_after, shuffle=False)
        all_loader = self._make_loader(ConcatDataset([base_dataset_after, inc_dataset_after]), shuffle=False)

        base_after = self._eval_loader(base_loader_after, mode, report=report)
        inc_after = self._eval_loader(inc_loader_after, mode, report=report)
        all_after = self._eval_loader(all_loader, mode, report=report)

        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "num_classes": len(self.label_map),
            "buffer_size": len(self.buffer),
            "base_before": base_before,
            "base_after": base_after,
            "incremental_after": inc_after,
            "all_after": all_after,
            "forgetting_score": forgetting_score(base_before["accuracy"], base_after["accuracy"]),
        }

    def evaluate_source(self, source, report: bool = False, use_prototypes: bool = False) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Model has not been created.")
        dataset = build_dataset(source, self.cfg, self.label_map)
        loader = self._make_loader(dataset, shuffle=False)
        return evaluate(
            self.model,
            loader,
            self.device,
            report=report,
            use_prototypes=use_prototypes,
            prototype_temperature=self.cfg.prd_temperature,
        )

    @torch.no_grad()
    def predict_loader(self, loader, use_prototypes: bool = False):
        if self.model is None:
            raise RuntimeError("Model has not been created.")
        from .metrics import predict
        return predict(self.model, loader, self.device, use_prototypes=use_prototypes, prototype_temperature=self.cfg.prd_temperature)

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
