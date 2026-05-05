import copy
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import CLConfig
from .data import MappedSubset
from .memory import ExemplarMemory
from .metrics import compute_forgetting_score, evaluate
from .models import ContinualResNet
from .trainer import build_optimizer, train_one_task
from .utils import (
    build_class_mapping,
    ensure_dir,
    get_device,
    get_indices_by_classes,
    get_labels_from_dataset,
    make_tasks,
    set_seed,
    task_names,
)


class ContinualLearner:
    """
    Library-style continual learner for image classification.

    Example:
        cfg = CLConfig(strategy="replay_kd")
        learner = ContinualLearner(cfg, class_names, in_channels=3)
        results = learner.fit(train_dataset, test_dataset)
    """
    def __init__(
        self,
        config: CLConfig,
        class_names: List[str],
        in_channels: int = 3,
        device: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        self.cfg = config
        self.cfg.validate()

        set_seed(self.cfg.seed)

        self.class_names = list(class_names)
        self.in_channels = int(in_channels)
        self.device = get_device(device)

        ensure_dir(self.cfg.output_dir)

        if model is None:
            self.model = ContinualResNet(
                in_channels=self.in_channels,
                num_classes=0,
                backbone_name=self.cfg.backbone,
                use_pretrained=self.cfg.use_pretrained,
                freeze_backbone=self.cfg.freeze_backbone,
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        self.teacher_model = None
        self.memory = ExemplarMemory(
            memory_per_class=self.cfg.memory_per_class,
            seed=self.cfg.seed,
        )

        self.tasks = make_tasks(
            num_classes=len(self.class_names),
            class_names=self.class_names,
            cfg=self.cfg,
        )

        self.class_to_idx = build_class_mapping(self.tasks)
        self.readable_tasks = task_names(self.tasks, self.class_names)

        self.acc_matrix = np.full((len(self.tasks), len(self.tasks)), np.nan)
        self.results = []
        self.seen_original_classes: List[int] = []

    def print_task_summary(self):
        print("
Continual tasks:")
        for i, task_classes in enumerate(self.tasks):
            print(
                f"Task {i}: class index {task_classes} | "
                f"class name {self.readable_tasks[i]}"
            )

    def fit(self, train_dataset, test_dataset, verbose: bool = True) -> pd.DataFrame:
        train_labels = get_labels_from_dataset(train_dataset)
        test_labels = get_labels_from_dataset(test_dataset)

        if verbose:
            print(f"Device        : {self.device}")
            print(f"Strategy      : {self.cfg.strategy}")
            print(f"Backbone      : {self.cfg.backbone}")
            print(f"Num classes   : {len(self.class_names)}")
            self.print_task_summary()

        for task_id, task_classes in enumerate(self.tasks):
            if verbose:
                print("
" + "=" * 70)
                print(f"START TASK {task_id}")
                print(f"New class index: {task_classes}")
                print(f"New class name : {[self.class_names[c] for c in task_classes]}")

            self.seen_original_classes.extend(task_classes)
            seen_mapped_classes = [
                self.class_to_idx[c] for c in self.seen_original_classes
            ]

            current_num_classes = len(self.seen_original_classes)
            self.model.expand_classifier(current_num_classes)

            if verbose:
                print(f"Classifier output classes: {self.model.num_classes}")

            current_train_indices = get_indices_by_classes(
                train_labels,
                task_classes,
            )

            current_train_dataset = MappedSubset(
                train_dataset,
                current_train_indices,
                self.class_to_idx,
            )

            memory_dataset = None
            if self.cfg.use_replay():
                memory_dataset = self.memory.get_dataset(
                    train_dataset,
                    self.class_to_idx,
                )

            if memory_dataset is not None:
                task_train_dataset = ConcatDataset([
                    current_train_dataset,
                    memory_dataset,
                ])

                if verbose:
                    print(f"Current task samples : {len(current_train_dataset)}")
                    print(f"Replay memory samples: {len(memory_dataset)}")
                    print(f"Total training samples: {len(task_train_dataset)}")
            else:
                task_train_dataset = current_train_dataset

                if verbose:
                    print(f"Current task samples : {len(current_train_dataset)}")
                    print("Replay memory samples: 0")
                    print(f"Total training samples: {len(task_train_dataset)}")

            train_loader = DataLoader(
                task_train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

            optimizer = build_optimizer(self.model, self.cfg)

            for epoch in range(1, self.cfg.epochs_per_task + 1):
                train_log = train_one_task(
                    model=self.model,
                    teacher_model=self.teacher_model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    device=self.device,
                    cfg=self.cfg,
                    task_id=task_id,
                )

                if verbose:
                    print(
                        f"Task {task_id} | Epoch {epoch}/{self.cfg.epochs_per_task} | "
                        f"Loss: {train_log['loss']:.4f} | "
                        f"CE: {train_log['ce_loss']:.4f} | "
                        f"KD: {train_log['kd_loss']:.4f}"
                    )

            if verbose:
                print("
Evaluation per task:")

            for eval_task_id in range(task_id + 1):
                eval_classes = self.tasks[eval_task_id]
                eval_indices = get_indices_by_classes(
                    test_labels,
                    eval_classes,
                )

                eval_dataset = MappedSubset(
                    test_dataset,
                    eval_indices,
                    self.class_to_idx,
                )

                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=(self.device.type == "cuda"),
                )

                eval_metric_labels = [self.class_to_idx[c] for c in eval_classes]

                metrics = evaluate(
                    model=self.model,
                    data_loader=eval_loader,
                    device=self.device,
                    metric_labels=eval_metric_labels,
                )

                self.acc_matrix[task_id, eval_task_id] = metrics["accuracy"]

                if verbose:
                    print(
                        f"After Task {task_id} | Test Task {eval_task_id} | "
                        f"Acc: {metrics['accuracy']:.4f} | "
                        f"Recall: {metrics['recall_macro']:.4f} | "
                        f"F1: {metrics['f1_macro']:.4f}"
                    )

                self.results.append({
                    "strategy": self.cfg.strategy,
                    "after_training_task": task_id,
                    "eval_scope": f"task_{eval_task_id}",
                    "eval_class_indices": str(eval_classes),
                    "eval_class_names": str([self.class_names[c] for c in eval_classes]),
                    "accuracy": metrics["accuracy"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "forgetting_score": np.nan,
                })

            seen_test_indices = get_indices_by_classes(
                test_labels,
                self.seen_original_classes,
            )

            seen_test_dataset = MappedSubset(
                test_dataset,
                seen_test_indices,
                self.class_to_idx,
            )

            seen_test_loader = DataLoader(
                seen_test_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

            seen_metrics = evaluate(
                model=self.model,
                data_loader=seen_test_loader,
                device=self.device,
                metric_labels=seen_mapped_classes,
            )

            forgetting_score = compute_forgetting_score(
                acc_matrix=self.acc_matrix,
                current_task=task_id,
            )

            if verbose:
                print("
Cumulative seen-class evaluation:")
                print(
                    f"After Task {task_id} | Seen Classes | "
                    f"Acc: {seen_metrics['accuracy']:.4f} | "
                    f"Recall: {seen_metrics['recall_macro']:.4f} | "
                    f"F1: {seen_metrics['f1_macro']:.4f} | "
                    f"Forgetting: {forgetting_score:.4f}"
                )

            self.results.append({
                "strategy": self.cfg.strategy,
                "after_training_task": task_id,
                "eval_scope": "seen_classes",
                "eval_class_indices": str(self.seen_original_classes),
                "eval_class_names": str([self.class_names[c] for c in self.seen_original_classes]),
                "accuracy": seen_metrics["accuracy"],
                "recall_macro": seen_metrics["recall_macro"],
                "f1_macro": seen_metrics["f1_macro"],
                "forgetting_score": forgetting_score,
            })

            if self.cfg.use_replay():
                self.memory.update(
                    labels=train_labels,
                    candidate_indices=current_train_indices,
                    classes=task_classes,
                )

                if verbose:
                    print(f"
Replay memory total samples after update: {len(self.memory)}")

            # Teacher is always stored, but it is only used when strategy uses KD.
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

            if self.cfg.save_checkpoint:
                self.save_checkpoint(task_id)

        results_df = pd.DataFrame(self.results)

        if self.cfg.save_results:
            self.save_results(results_df)

        return results_df

    def save_checkpoint(self, task_id: int):
        ckpt_path = os.path.join(
            self.cfg.output_dir,
            f"checkpoint_after_task_{task_id}.pt",
        )

        torch.save({
            "task_id": task_id,
            "strategy": self.cfg.strategy,
            "model_state_dict": self.model.state_dict(),
            "num_classes": self.model.num_classes,
            "seen_original_classes": self.seen_original_classes,
            "seen_class_names": [self.class_names[c] for c in self.seen_original_classes],
            "class_to_idx": self.class_to_idx,
            "class_names": self.class_names,
            "tasks": self.tasks,
            "cfg": self.cfg.__dict__,
        }, ckpt_path)

    def save_results(self, results_df: pd.DataFrame):
        results_path = os.path.join(
            self.cfg.output_dir,
            "continual_learning_results.csv",
        )

        acc_matrix_path = os.path.join(
            self.cfg.output_dir,
            "accuracy_matrix.npy",
        )

        results_df.to_csv(results_path, index=False)
        np.save(acc_matrix_path, self.acc_matrix)
