from typing import Dict, Any, Union, Callable
from os import PathLike
import os
from math import ceil
import json
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from ignite.engine import (
    Events,
    _prepare_batch,
    create_supervised_evaluator,
)
from ignite.handlers import global_step_from_engine, Checkpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar, WandBLogger
import wandb
from utils import (
    parse_log_interval,
    create_object_from_config,
    create_lr_scheduler_from_config,
    predict_test_images,
)
from engines.supervised import create_supervised_trainer
from losses.composite import CompositeLoss


def _fix_metric_dtypes(data):
    """
    Ensure output output from evaluator has same dtype and device as input.
    """
    y_pred, y = data
    y_pred = torch.as_tensor(y_pred, dtype=y.dtype, device=y.device)
    return y_pred, y


def train(
    config: Dict[str, Any],
    device: Union[str, torch.device],
    checkpoint: Union[str, PathLike] = None,
    logging: str = "online",
    prepare_batch: Callable = _prepare_batch,
    trainer_model_transform: Callable[[Any], Any] = lambda output: output,
    trainer_output_transform: Callable[
        [Any, Any, Any, torch.Tensor], Any
    ] = lambda x, y, y_pred, loss: loss.item(),
    evaluator_model_transform: Callable[[Any], Any] = lambda output: output,
    evaluator_output_transform: Callable[
        [Any, Any, Any], Any
    ] = lambda x, y, y_pred: (y_pred, y),
):
    device = torch.device(device)

    # Parse config file
    amp = config.get("amp", False)
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
    loss_labels = []
    loss_modules = []
    loss_weights = []
    for loss_label, loss_conf in config["losses"].items():
        loss_labels.append(loss_label)
        loss_modules.append(create_object_from_config(loss_conf))
        loss_weights.append(loss_conf["weight"])

    loss_fn = CompositeLoss(loss_labels, loss_modules, loss_weights).to(device)

    # Create metrics
    metrics = {}
    for metric_label, metric_conf in config["metrics"].items():
        metric = create_object_from_config(
            config=metric_conf,
            output_transform=_fix_metric_dtypes,
        )
        metrics[metric_label] = metric

    # Create trainer
    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        amp=amp,
        scaler=amp,
        clip_grad_norm=clip_grad_norm,
        prepare_batch=prepare_batch,
        model_transform=trainer_model_transform,
        output_transform=trainer_output_transform,
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
    evaluator = create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        amp_mode="amp" if amp else None,
        prepare_batch=prepare_batch,
        model_transform=evaluator_model_transform,
        output_transform=evaluator_output_transform,
    )
    ProgressBar(desc="Val", ncols=80).attach(evaluator)

    # Set up logging
    wandb_logger = WandBLogger(
        mode=logging,
        project=config["project"],
        config=dict(
            config=config,
            num_params=sum(p.numel() for p in model.parameters()),
        ),
    )

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
        labels = ["loss/" + l for l in loss_fn.labels]
        losses = dict(zip(labels, loss_fn.last_values))
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
            prepare_batch=prepare_batch,
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

    if logging != "disabled":
        run_name = (
            wandb_logger.run.name
            if logging == "online"
            else f"offline-{wandb_logger.run.id}"
        )

        checkpoint_handler = Checkpoint(
            to_save=to_save,
            save_handler=f"checkpoints/{run_name}",
            filename_prefix="best",
            score_name=config["checkpoint_metric"],
            n_saved=3,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        with open(f"checkpoints/{run_name}/config.json", "w") as fp:
            json.dump(config, fp, indent=2)

    # Start training
    if checkpoint:
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    trainer.run(loaders["train"], max_epochs=max_epochs)
    wandb_logger.close()
