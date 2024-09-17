from typing import Callable, Any, Union, Sequence, Tuple, Optional
import torch
from ignite.engine import Engine, DeterministicEngine
from accelerate import Accelerator


def create_gan_trainer(
    accelerator: Accelerator,
    model: torch.nn.Module,
    disc_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    disc_scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: Union[Callable, torch.nn.Module],
    disc_loss_fn: Union[Callable, torch.nn.Module],
    clip_grad_norm: Optional[float] = None,
    clip_grad_value: Optional[float] = None,
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
    scheduler : torch LRScheduler
        Model learning rate scheduler.
    disc_scheduler : torch LRScheduler
        Discriminator learning rate scheduler.
    loss_fn : torch.nn.Module
        The supervised loss function to use for model.
    disc_loss_fn : torch.nn.Module
        The loss function to use discriminator.
    clip_grad_norm : float
        Clip gradients to norm if provided.
    update_interval : int
        How many steps between updating `model`.
    disc_update_interval : int
        How many steps between updating `disc_model`.
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

    Returns
    -------
    trainer : torch.engine.Engine
        A trainer engine with GAN update function.
    """

    def _update(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        disc_model.train()

        x, y = batch
        x, y = input_transform(x, y)

        # Update discriminator
        with accelerator.accumulate(disc_model):
            y_pred = model_transform(model(x)).detach()
            disc_pred_real = disc_model_transform(disc_model(y))
            disc_pred_fake = disc_model_transform(disc_model(y_pred))
            disc_loss = disc_loss_fn(
                y_pred,
                y,
                disc_real=disc_pred_real,
                disc_fake=disc_pred_fake,
            )

            accelerator.backward(disc_loss)
            if accelerator.sync_gradients:
                if clip_grad_norm:
                    accelerator.clip_grad_norm_(
                        parameters=disc_model.parameters(),
                        max_norm=clip_grad_norm,
                    )
                if clip_grad_value:
                    accelerator.clip_grad_value_(
                        parameters=disc_model.parameters(),
                        clip_value=clip_grad_value,
                    )

            disc_optimizer.step()
            disc_scheduler.step()
            disc_optimizer.zero_grad()

        # Update generator
        with accelerator.accumulate(model):
            y_pred = model_transform(model(x))
            disc_pred_fake = disc_model_transform(disc_model(y_pred))
            loss = loss_fn(
                y_pred,
                y,
                disc_fake=disc_pred_fake,
            )

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
            optimizer.zero_grad()

        return output_transform(x, y, y_pred, loss, disc_loss)

    trainer = (
        Engine(_update) if not deterministic else DeterministicEngine(_update)
    )

    return trainer
