from typing import Union, Callable, Any, Sequence, Tuple, Optional, Dict
import torch
from ignite.engine import Engine, DeterministicEngine
from ignite.metrics import Metric
from accelerate import Accelerator


def create_supervised_trainer(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: Union[Callable, torch.nn.Module],
    clip_grad_norm: Optional[float] = None,
    clip_grad_value: Optional[float] = None,
    input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
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
    model : torch.nn.Module
        The model to train.
    optimizer : torch optimizer
        The optimizer to use.
    scheduler : torch LRScheduler
        Learning rate scheduler.
    loss_fn : torch.nn.Module
        The loss function to use.
    clip_grad_norm : float
        Clip gradients to norm if provided.
    clip_grad_value : float
        Clip gradients to value if provided.
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

    Returns
    -------
    trainer : torch.ignite.Engine
        A trainer engine with supervised update function.
    """

    def _update(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()

        x, y = batch
        x, y = input_transform(x, y)

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            y_pred = model_transform(model(x))
            loss = loss_fn(y_pred, y)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if clip_grad_norm:
                    accelerator.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=clip_grad_norm,
                    )
                if clip_grad_value:
                    accelerator.clip_grad_value_(
                        parameters=model.parameters(),
                        clip_value=clip_grad_value,
                    )

            optimizer.step()
            scheduler.step()

        return output_transform(x, y, y_pred, loss)

    trainer = (
        Engine(_update) if not deterministic else DeterministicEngine(_update)
    )

    return trainer


def create_supervised_evaluator(
    accelerator: Accelerator,
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
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

    def _step(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()

        x, y = batch
        x, y = input_transform(x, y)
        with torch.no_grad():
            y_pred = model_transform(model(x))

        x, y, y_pred = accelerator.gather_for_metrics((x, y, y_pred))
        return output_transform(x, y, y_pred)

    evaluator = Engine(_step)

    if metrics:
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    return evaluator
