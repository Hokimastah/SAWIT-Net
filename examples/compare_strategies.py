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
        output_dir=f"outputs_compare_{strategy}",
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
