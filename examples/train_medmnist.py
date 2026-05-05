from sawit_net import CLConfig, ContinualLearner
from sawit_net.data import load_medmnist_dataset


train_dataset, test_dataset, class_names, channels = load_medmnist_dataset(
    medmnist_name="organcmnist",
    img_size=128,
    input_mode="rgb",
)

cfg = CLConfig(
    strategy="kd_only",        # KD only
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
print(results.tail())
