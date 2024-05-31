from typing import Union, Callable, Any, Sequence, Tuple, Optional, Dict
import torch
from ignite.engine import Engine, DeterministicEngine, Events, _prepare_batch
from ignite.handlers import TerminateOnNan
from ignite.metrics import Metric
from .common import _backward_with_scaler


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Union[str, torch.device] = "cpu",
    amp: bool = False,
    clip_grad_norm: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[
        [Any, Any, Any, torch.Tensor], Any
    ] = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    terminate_on_nan: bool = True,
) -> Engine:
    """
    Factory function for supervised trainer.

    Parameters
    ----------
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
    amp : bool
        If `true` automatic mixed-precision is enabled.
    clip_grad_norm : float
        Clip gradients to norm if provided.
    gradient_accumulation_steps : int
        Number of steps the gradients should be accumulated across.
    non_blocking : bool
        If `True` and this copy is between CPU and GPU, the copy may
        occur asynchronously with respect to the host.
        For other cases, this argument has no effect.
    prepare_batch : Callable
        Function that receives `batch`, `device`, `non_blocking`
        and outputs tuple of tensors `(batch_x, batch_y)`.
    input_transform : Callable
        Function that receives tensors `y` and `y` and outputs tuple of
        tensors `(x, y)`.
    model_transform : Callable
        Function that receives the output from the model and
        convert it into the form as required by the loss function.
    output_transform : Callable
        Function that receives `x`, `y`, `y_pred`, `loss` and
        returns value to be assigned to engine's state.output after each
        iteration. Default is returning `loss.item()`.
    deterministic : bool
        If `True`, returns `DeterministicEngine`, otherwise `Engine`.
    terminate_on_nan: bool
        Terminate training if model outputs NaN or infinite.

    Returns
    -------
    trainer : torch.ignite.Engine
        A trainer engine with supervised update function.
    """
    device = torch.device(device)
    if "xla" in device.type:
        raise ValueError("TPU not supported in trainer.")

    scaler = None
    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    def _update(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        x, y = input_transform(x, y)

        with torch.autocast(device_type=device.type, enabled=amp):
            y_pred = model_transform(model(x))
            loss = loss_fn(y_pred, y)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

        _backward_with_scaler(
            model=model,
            optimizer=optimizer,
            loss=loss,
            iteration=engine.state.iteration,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clip_grad_norm=clip_grad_norm,
            scaler=scaler,
        )

        return output_transform(
            x, y, y_pred, loss * gradient_accumulation_steps
        )

    trainer = (
        Engine(_update) if not deterministic else DeterministicEngine(_update)
    )

    if terminate_on_nan:
        trainer.add_event_handler(
            event_name=Events.ITERATION_COMPLETED,
            handler=TerminateOnNan(),
        )

    return trainer


def create_supervised_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Union[str, torch.device] = "cpu",
    amp: bool = False,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
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
    model : torch.nn.Module
        The model to train.
    metrics : dict
        Dictionary of evaluation metrics.
    device : torch.device
        Device type specification.
        Applies to batches after starting the engine. Model will not be moved.
        Device can be CPU, GPU.
    amp : bool
        If `true` automatic mixed-precision is enabled.
    non_blocking : bool
        If `True` and this copy is between CPU and GPU, the copy may
        occur asynchronously with respect to the host.
        For other cases, this argument has no effect.
    prepare_batch : Callable
        Function that receives `batch`, `device`, `non_blocking`
        and outputs tuple of tensors `(batch_x, batch_y)`.
    input_transform : Callable
        Function that receives tensors `y` and `y` and outputs tuple of
        tensors `(x, y)`.
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
            x, y = input_transform(x, y)

            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(x)
                y_pred = model_transform(output)

            return output_transform(x, y, y_pred)

    evaluator = Engine(_step)

    if metrics:
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    return evaluator
