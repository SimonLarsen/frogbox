from typing import Union, Callable, Any, Sequence, Tuple, Optional, Dict
import torch
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine import Engine, _prepare_batch
from ignite.metrics import Metric
from ..config import Config


def create_supervised_trainer(
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Union[str, torch.device] = "cpu",
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[
        [Any, Any, Any, torch.Tensor], Any
    ] = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:
    """
    Factory function for supervised trainer.

    Parameters
    ----------
    config : Config
        Project configuration.
    model : torch.nn.Module
        The model to train.
    optimizer : torch optimizer
        The optimizer to use.
    loss_fn : torch.nn.Module
        The loss function to use.
    device : torch.device
        Device type specification.
        Applies to batches after starting the engine. Model will not be moved.
        Device can be CPU, GPU.
    non_blocking : bool
        If `True` and this copy is between CPU and GPU, the copy may
        occur asynchronously with respect to the host.
        For other cases, this argument has no effect.
    prepare_batch : Callable
        Function that receives `batch`, `device`, `non_blocking`
        and outputs tuple of tensors `(batch_x, batch_y)`.
    model_transform : Callable
        Function that receives the output from the model and
        convert it into the form as required by the loss function.
    output_transform : Callable
        Function that receives `x`, `y`, `y_pred`, `loss` and
        returns value to be assigned to engine's state.output after each
        iteration. Default is returning `loss.item()`.
    deterministic : bool
        If `True`, returns `DeterministicEngine`, otherwise `Engine`.

    Returns
    -------
    trainer : torch.ignite.Engine
        A trainer engine with supervised update function.
    """
    amp = config.amp
    clip_grad_norm = config.clip_grad_norm
    gradient_accumulation_steps = config.gradient_accumulation_steps
    scaler = None

    device = torch.device(device)
    if "xla" in device.type:
        raise ValueError("TPU not supported in trainer.")
    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    def _update(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(x)
            y_pred = model_transform(output)
            loss = loss_fn(y_pred, y)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
            if engine.state.iteration % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=clip_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if engine.state.iteration % gradient_accumulation_steps == 0:
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=clip_grad_norm,
                    )
                optimizer.step()

        return output_transform(
            x, y, y_pred, loss * gradient_accumulation_steps
        )

    trainer = (
        Engine(_update) if not deterministic else DeterministicEngine(_update)
    )
    return trainer


def create_supervised_evaluator(
    config: Config,
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Union[str, torch.device] = "cpu",
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (
        y_pred,
        y,
    ),
) -> Engine:
    """
    Factory function for supervised evaluator.

    Parameters
    ----------
    config : Config
        Project configuration.
    model : torch.nn.Module
        The model to train.
    metrics : dict
        Dictionary of evaluation metrics.
    device : torch.device
        Device type specification.
        Applies to batches after starting the engine. Model will not be moved.
        Device can be CPU, GPU.
    non_blocking : bool
        If `True` and this copy is between CPU and GPU, the copy may
        occur asynchronously with respect to the host.
        For other cases, this argument has no effect.
    prepare_batch : Callable
        Function that receives `batch`, `device`, `non_blocking`
        and outputs tuple of tensors `(batch_x, batch_y)`.
    model_transform : Callable
        Function that receives the output from the model and convert it into
        the predictions: `y_pred = model_transform(model(x))`.
    output_transform : Callable
        Function that receives `x`, `y`, `y_pred` and returns value to be
        assigned to engine's state.output after each iteration.
        Default is returning `(y_pred, y,)` which fits output expected by
        metrics. If you change it you should use `output_transform` in metrics.
    """

    device = torch.device(device)
    if "xla" in device.type:
        raise ValueError("TPU not supported in trainer.")

    def _step(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(
                batch, device=device, non_blocking=non_blocking
            )
            with torch.autocast(device_type=device.type, enabled=config.amp):
                output = model(x)
                y_pred = model_transform(output)
            return output_transform(x, y, y_pred)

    evaluator = Engine(_step)

    if metrics:
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    return evaluator
