import os
import json
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Average, RunningAverage, SSIM, PSNR
from ignite.handlers import global_step_from_engine
from ignite.handlers.checkpoint import Checkpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
import wandb
from utils import create_object_from_config, predict_test_images
from typing import Optional, Sequence


def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args(args)


args = parse_arguments()

with open(args.config, "r") as fp:
    config = json.load(fp)

amp = config.get("amp", False)
device = torch.device(args.device)
batch_size = config["batch_size"]
loader_workers = config.get("loader_workers", 0)

# Create data loaders
datasets = {}
loaders = {}
for split, ds_conf in config["datasets"].items():
    ds = create_object_from_config(ds_conf)
    datasets[split] = ds
    loaders[split] = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=loader_workers,
        shuffle=split == "train",
    )

# Create model
model = create_object_from_config(config["model"])
model = model.to(device)
optimizer = create_object_from_config(
    config=config["optimizer"],
    params=model.parameters(),
)

# Create loss function
loss_fn = torch.nn.L1Loss()

# Create trainer
trainer = create_supervised_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    amp_mode="amp" if amp else None,
    scaler=amp,
)
Average().attach(trainer, "loss")
RunningAverage(alpha=0.5, output_transform=lambda x: x).attach(
    trainer, "running_avg_loss"
)
ProgressBar(desc="Train", ncols=80).attach(trainer, ["running_avg_loss"])

# Create learning rate scheduler
lr_scheduler = create_object_from_config(
    config=config["lr_scheduler"],
    optimizer=optimizer,
    param_name="lr",
)
trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)


# Create evaluator
def fix_metric_dtypes(data):
    y_pred, y = data
    y_pred = torch.as_tensor(y_pred, dtype=y.dtype, device=y.device)
    return y_pred, y


metrics = {
    "ssim": SSIM(data_range=1.0, output_transform=fix_metric_dtypes),
    "psnr": PSNR(data_range=1.0, output_transform=fix_metric_dtypes),
}
evaluator = create_supervised_evaluator(
    model=model,
    metrics=metrics,
    device=device,
    amp_mode="amp" if amp else None,
)
ProgressBar(desc="Val", ncols=80).attach(evaluator)

# Set up logging
wandb_logger = WandBLogger(
    mode=args.wandb_mode,
    project="example-project",
    config=dict(config=config),
)
run_name = wandb_logger.run.name

wandb_logger.attach_output_handler(
    engine=trainer,
    event_name=Events.EPOCH_COMPLETED,
    tag="train",
    metric_names=["loss"],
)

wandb_logger.attach_output_handler(
    engine=evaluator,
    event_name=Events.COMPLETED,
    tag="val",
    metric_names="all",
    global_step_transform=global_step_from_engine(trainer),
)

wandb_logger.attach_opt_params_handler(
    engine=trainer,
    event_name=Events.EPOCH_COMPLETED,
    optimizer=optimizer,
    param_name="lr",
)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation(trainer):
    evaluator.run(loaders["val"])


@trainer.on(Events.EPOCH_COMPLETED)
def log_test_images(trainer):
    def output_transform(x, y, y_pred):
        if datasets["test"].do_normalize:
            x = datasets["test"].denormalize(x)
        return x, y_pred, y

    images = predict_test_images(
        model=model,
        data=loaders["test"],
        device=device,
        output_transform=output_transform,
        resize_to_fit=True,
    )

    images = [wandb.Image(to_pil_image(image)) for image in images]
    wandb.log(step=trainer.state.epoch, data={"test/images": images})


# Set up checkpoints
to_save = {
    "model": model,
    "optimizer": optimizer,
    "trainer": trainer,
    "lr_scheduler": lr_scheduler,
}
checkpoint_handler = Checkpoint(
    to_save=to_save,
    save_handler=f"checkpoints/{run_name}",
    filename_prefix="best",
    score_name="psnr",
    n_saved=3,
    global_step_transform=global_step_from_engine(trainer),
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
with open(f"checkpoints/{run_name}/config.json", "w") as fp:
    json.dump(config, fp, indent=2)

# Start training
if args.checkpoint:
    Checkpoint.load_objects(to_load=to_save, checkpoint=args.checkpoint)

trainer.run(
    loaders["train"],
    max_epochs=config["max_epochs"],
    epoch_length=config.get("epoch_length"),
)
wandb_logger.close()
