import torch
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from ignite.engine import _prepare_batch
from typing import Union, Callable, Any, Optional, Sequence, Tuple


def create_supervised_trainer(
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
    amp: bool = False,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
) -> Engine:
    """
    Factory function for supervised evaluation.

    Similar to ignite.engine.create_supervised_trainer except:
    1. Gradient clipping is added through the `clip_grad_norm` argument.
    2. TPU and APEX is not currently supported.
       AMP is enabled with the `amp` argument.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    optimizer : torch optimizer
        The optimizer to use.
    loss_fn : torch.nn.Module
        The loss function to use.
    device : torch.device
        Device type specification (default: None).
        Applies to batches after starting the engine. Model will not be moved.
        Device can be CPU, GPU or TPU.
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
        Function that receives 'x', 'y', 'y_pred', 'loss' and
        returns value to be assigned to engine's state.output after each
        iteration. Default is returning `loss.item()`.
    deterministic : bool
        If `True`, returns `DeterministicEngine`, otherwise `Engine`.
    amp : bool
        If `True`, enables automatic mixed-precision.
    scaler : torch.cuda.amp.GradScaler
        GradScaler instance for gradient scaling. If `True`, will create
        default GradScaler. If GradScaler instance is passed, it will be used.
    gradient_accumulation_steps : int
        Number of steps to accumulate gradients over.
    clip_grad_norm : float
        Clip gradients to norm if provided.

    Returns
    -------
    trainer : torch.ignite.Engine
        A trainer engine with supervised update function.
    """
    device = torch.device(device)
    device_type = device.type if isinstance(device, torch.device) else device
    if "xla" in device_type:
        raise ValueError("TPU not supported in trainer.")
    if amp and isinstance(scaler, bool) and scaler:
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
    if scaler:
        trainer.state.scaler = scaler
    return trainer
