from typing import Callable, Any, Union, Sequence, Tuple, Optional
import torch
from ignite.engine import Engine, DeterministicEngine, Events, _prepare_batch
from ignite.handlers import TerminateOnNan
from .common import _backward_with_scaler


def create_gan_trainer(
    model: torch.nn.Module,
    disc_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    disc_loss_fn: Union[Callable, torch.nn.Module],
    device: Union[str, torch.device] = "cpu",
    amp: bool = False,
    clip_grad_norm: Optional[float] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
    model_transform: Callable[[Any], Any] = lambda output: output,
    disc_model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[
        [Any, Any, Any, torch.Tensor, torch.Tensor], Any
    ] = lambda x, y, y_pred, loss, disc_loss: (
        loss.item(),
        disc_loss.item(),
    ),
    deterministic: bool = False,
    terminate_on_nan: bool = True,
) -> Engine:
    """
    Factory function for GAN trainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    disc_model : torch.nn.Module
        The discriminator to train.
    optimizer : torch optimizer
        The optimizer to use for model.
    disc_optimizer : torch optimizer
        The optimizer to use for discriminator.
    loss_fn : torch.nn.Module
        The supervised loss function to use for model.
    disc_loss_fn : torch.nn.Module
        The loss function to use discriminator.
    device : torch.device
        Device type specification.
        Applies to batches after starting the engine. Model will not be moved.
        Device can be CPU, GPU.
    amp : bool
        If `true` automatic mixed-precision is enabled.
    clip_grad_norm : float
        Clip gradients to norm if provided.
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
    disc_model_transform : Callable
        Function that receives the output from the discriminator and
        convert it into the form as required by the loss function.
    output_transform : Callable
        Function that receives `x`, `y`, `y_pred`, `loss` and returns value
        to be assigned to engine's state.output after each iteration. Default
        is returning `(loss.item(), disc_loss.item())`.
    deterministic : bool
        If `True`, returns `DeterministicEngine`, otherwise `Engine`.
    terminate_on_nan: bool
        Terminate training if model outputs NaN or infinite.

    Returns
    -------
    trainer : torch.engine.Engine
        A trainer engine with GAN update function.
    """
    device = torch.device(device)
    if "xla" in device.type:
        raise ValueError("TPU not supported in trainer.")

    scaler = None
    disc_scaler = None
    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        disc_scaler = torch.cuda.amp.GradScaler(enabled=True)

    def _update(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        disc_model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        x, y = input_transform(x, y)

        # Update discriminator
        disc_optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=amp):
            y_pred = model_transform(model(x))
            disc_pred_real = disc_model_transform(disc_model(y))
            disc_pred_fake = disc_model_transform(disc_model(y_pred.detach()))
            disc_loss = disc_loss_fn(
                y_pred, y, disc_real=disc_pred_real, disc_fake=disc_pred_fake
            )

        _backward_with_scaler(
            model=disc_model,
            optimizer=disc_optimizer,
            loss=disc_loss,
            iteration=engine.state.iteration,
            clip_grad_norm=clip_grad_norm,
            scaler=disc_scaler,
        )

        # Update generator
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=amp):
            disc_pred_fake = disc_model_transform(disc_model(y_pred))
            loss = loss_fn(
                y_pred, y, disc_real=disc_pred_real, disc_fake=disc_pred_fake
            )

        _backward_with_scaler(
            model=model,
            optimizer=optimizer,
            loss=loss,
            iteration=engine.state.iteration,
            clip_grad_norm=clip_grad_norm,
            scaler=scaler,
        )

        return output_transform(x, y, y_pred, loss, disc_loss)

    trainer = (
        Engine(_update) if not deterministic else DeterministicEngine(_update)
    )

    if terminate_on_nan:
        trainer.add_event_handler(
            event_name=Events.ITERATION_COMPLETED,
            handler=TerminateOnNan(),
        )

    return trainer
