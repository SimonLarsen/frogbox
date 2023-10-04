import os
from math import ceil
import json
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from ignite.engine import (
    Events,
    create_supervised_evaluator,
)
from ignite.metrics import SSIM, PSNR
from ignite.handlers import global_step_from_engine, Checkpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar, WandBLogger
import wandb
from utils import (
    create_object_from_config,
    create_lr_scheduler_from_config,
    parse_log_interval,
    predict_test_images,
)
from engines.supervised import create_supervised_trainer
from losses.composite import CompositeLoss
from metrics.element_mae import ElementMeanAbsoluteError
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

project = config.get("project", Path(__file__).parent.name)
amp = config.get("amp", False)
device = torch.device(args.device)
batch_size = config["batch_size"]
loader_workers = config.get("loader_workers", 0)
max_epochs = config["max_epochs"]
clip_grad_norm = config.get("clip_grad_norm", None)
log_interval = Events.EPOCH_COMPLETED
if "log_interval" in config:
    log_interval = parse_log_interval(config["log_interval"])

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
loss_modules = []
loss_weights = []
for loss_conf in config["losses"]:
    loss_modules.append(create_object_from_config(loss_conf))
    loss_weights.append(loss_conf["weight"])

loss_fn = CompositeLoss(loss_modules, loss_weights).to(device)

# Create trainer
trainer = create_supervised_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    amp=amp,
    scaler=amp,
    clip_grad_norm=clip_grad_norm,
)
ProgressBar(desc="Train", ncols=80).attach(trainer)

# Create learning rate scheduler
max_iterations = ceil(len(datasets["train"]) / batch_size * max_epochs)
lr_scheduler = create_lr_scheduler_from_config(
    optimizer=optimizer,
    config=config["lr_scheduler"],
    max_iterations=max_iterations,
)
trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


# Create evaluator
def fix_metric_dtypes(data):
    y_pred, y = data
    y_pred = torch.as_tensor(y_pred, dtype=y.dtype, device=y.device)
    return y_pred, y


metrics = {
    "mae": ElementMeanAbsoluteError(output_transform=fix_metric_dtypes),
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
num_params = sum(p.numel() for p in model.parameters())
wandb_logger = WandBLogger(
    mode=args.wandb_mode,
    project=project,
    config=dict(config=config, num_params=num_params),
)
run_name = wandb_logger.run.name

wandb_logger.attach_output_handler(
    engine=trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="train",
    output_transform=lambda loss: {"loss": loss},
)

wandb_logger.attach_opt_params_handler(
    engine=trainer,
    event_name=Events.ITERATION_COMPLETED,
    optimizer=optimizer,
    param_name="lr",
)

wandb_logger.attach_output_handler(
    engine=evaluator,
    event_name=Events.COMPLETED,
    tag="val",
    metric_names="all",
    global_step_transform=global_step_from_engine(
        trainer, Events.ITERATION_COMPLETED
    ),
)


@trainer.on(Events.ITERATION_COMPLETED)
def log_losses(trainer):
    names = ["loss/" + l.get("name", l["class"]) for l in config["losses"]]
    losses = dict(zip(names, loss_fn.last_values))
    wandb.log(step=trainer.state.iteration, data=losses)


@trainer.on(log_interval)
def log_validation():
    evaluator.run(loaders["val"])


@trainer.on(log_interval)
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
    wandb.log(step=trainer.state.iteration, data={"test/images": images})


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

trainer.run(loaders["train"], max_epochs=max_epochs)
wandb_logger.close()
