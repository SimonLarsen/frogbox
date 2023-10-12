from typing import Optional, Any, Union, Callable, Sequence
from os import PathLike
import os
from math import ceil
import json
import torch
from torch.utils.data import DataLoader
from ignite.engine import (
    Events,
    _prepare_batch,
    create_supervised_evaluator,
)
from ignite.handlers import global_step_from_engine, Checkpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar, WandBLogger
import wandb
from ..config import (
    create_object_from_config,
    parse_log_interval,
    create_lr_scheduler_from_config,
)
from ..config import Config, CheckpointMode
from ..engines.supervised import create_supervised_trainer
from ..losses.composite import CompositeLoss
from ..callbacks.callback import Callback, CallbackState


def _fix_metric_dtypes(data):
    """
    Ensure output output from evaluator has same dtype and device as input.
    """
    y_pred, y = data
    y_pred = torch.as_tensor(y_pred, dtype=y.dtype, device=y.device)
    return y_pred, y


def train_supervised(
    config: Config,
    device: Union[str, torch.device],
    checkpoint: Optional[Union[str, PathLike]] = None,
    logging: str = "online",
    callbacks: Sequence[Callback] = None,
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
    """
    Train supervised model.

    Parameters
    ----------
    config : Config
        Pipeline configuration.
    device : torch.device
        CUDA device. Can be CPU or GPU. Model will not be moved.
    checkpoint : path-like
        Path to experiment checkpoint.
    logging : str
        Logging mode. Must be either "online", "offline" or "disabled".
    prepare_batch : Callable
        function that receives `batch`, `device`, `non_blocking` and outputs
        tuple of tensors `(batch_x, batch_y)`.
    trainer_model_transform : Callable
        function that receives the output from the model during training and
        converts it into the form as required by the loss function.
    trainer_output_transform : Callable
        function that receives `x`, `y`, `y_pred`, `loss` and returns value
        to be assigned to trainer's `state.output` after each iteration.
        Default is returning `loss.item()`.
    evaluator_model_transform : Callable
        function that receives the output from the model during evaluation and
        convert it into the predictions:
        ``y_pred = model_transform(model(x))``.
    evaluator_output_transform : Callable
        function that receives `x`, `y`, `y_pred` and returns value to be
        assigned to evaluator's `state.output` after each iteration.
        Default is returning `(y_pred, y,)` which fits output expected by
        metrics. If you change it you should use `output_transform` in metrics.
    """
    device = torch.device(device)

    # Parse config file
    log_interval = parse_log_interval(config.log_interval)

    # Create data loaders
    datasets = {}
    loaders = {}
    for split, ds_conf in config.datasets.items():
        ds = create_object_from_config(ds_conf)
        datasets[split] = ds
        loaders[split] = DataLoader(
            dataset=ds,
            batch_size=config.batch_size,
            num_workers=config.loader_workers,
            shuffle=split == "train",
        )

    # Create model
    model = create_object_from_config(config.model)
    model = model.to(device)
    optimizer = create_object_from_config(
        config=config.optimizer,
        params=model.parameters(),
    )

    # Create loss function
    loss_labels = []
    loss_modules = []
    loss_weights = []
    for loss_label, loss_conf in config.losses.items():
        loss_labels.append(loss_label)
        loss_modules.append(create_object_from_config(loss_conf))
        loss_weights.append(loss_conf.weight)

    loss_fn = CompositeLoss(loss_labels, loss_modules, loss_weights).to(device)

    # Create metrics
    metrics = {}
    for metric_label, metric_conf in config.metrics.items():
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
        amp=config.amp,
        scaler=config.amp,
        clip_grad_norm=config.clip_grad_norm,
        prepare_batch=prepare_batch,
        model_transform=trainer_model_transform,
        output_transform=trainer_output_transform,
    )
    ProgressBar(desc="Train", ncols=80).attach(trainer)

    # Create learning rate scheduler
    max_iterations = ceil(
        len(datasets["train"]) / config.batch_size * config.max_epochs
    )
    lr_scheduler = create_lr_scheduler_from_config(
        optimizer=optimizer,
        config=config.lr_scheduler,
        max_iterations=max_iterations,
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Create evaluator
    evaluator = create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        amp_mode="amp" if config.amp else None,
        prepare_batch=prepare_batch,
        model_transform=evaluator_model_transform,
        output_transform=evaluator_output_transform,
    )
    ProgressBar(desc="Val", ncols=80).attach(evaluator)

    # Set up logging
    wandb_logger = WandBLogger(
        mode=logging,
        project=config.project,
        config=dict(
            config=config.model_dump(),
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
        labels = ["loss/" + label for label in loss_fn.labels]
        losses = dict(zip(labels, loss_fn.last_values))
        wandb.log(step=trainer.state.iteration, data=losses)

    @trainer.on(log_interval)
    def log_validation():
        evaluator.run(loaders["val"])

    # Add callback functions
    if callbacks:
        for callback in callbacks:

            def _callback_handler():
                state = CallbackState(
                    trainer=trainer,
                    evaluator=evaluator,
                    datasets=datasets,
                    loaders=loaders,
                    model=model,
                    config=config,
                    device=device,
                )
                callback.function(state)

            trainer.add_event_handler(callback.event, _callback_handler)

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

        score_fn = Checkpoint.get_default_score_fn(
            metric_name=config.checkpoint_metric,
            score_sign=(
                1.0 if config.checkpoint_mode == CheckpointMode.MAX else -1.0
            ),
        )

        checkpoint_handler = Checkpoint(
            to_save=to_save,
            save_handler=f"checkpoints/{run_name}",
            filename_prefix="best",
            score_name=config.checkpoint_metric,
            score_function=score_fn,
            n_saved=config.checkpoint_n_saved,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        with open(f"checkpoints/{run_name}/config.json", "w") as fp:
            json.dump(config.model_dump(), fp, indent=2)

    # Start training
    if checkpoint:
        Checkpoint.load_objects(to_load=to_save, checkpoint=str(checkpoint))

    trainer.run(loaders["train"], max_epochs=config.max_epochs)
    wandb_logger.close()
