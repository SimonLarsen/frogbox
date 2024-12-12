from typing import Callable, Any, Optional
import torch
from accelerate import Accelerator
from .engine import Engine


class GANTrainer(Engine):
    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        disc_model: torch.nn.Module,
        loss_fn: Callable[..., Any],
        disc_loss_fn: Callable[..., Any],
        optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        disc_scheduler: torch.optim.lr_scheduler.LRScheduler,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
        model_transform: Callable[[Any], Any] = lambda output: output,
        disc_model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[
            [Any, Any, Any, Any, Any], Any
        ] = lambda x, y, y_pred, loss, disc_loss: (
            loss.item(),
            disc_loss.item(),
        ),
        **kwargs,
    ):
        self.accelerator = accelerator
        self.model = model
        self.disc_model = disc_model
        self.loss_fn = loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.scheduler = scheduler
        self.disc_scheduler = disc_scheduler

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self._input_transform = input_transform
        self._model_transform = model_transform
        self._disc_model_transform = disc_model_transform
        self._output_transform = output_transform

        super().__init__(self.process, **kwargs)

    def process(self, batch):
        self.model.train()
        self.disc_model.train()

        inputs, targets = batch
        inputs, targets = self._input_transform(inputs, targets)

        # Update discriminator
        with self.accelerator.accumulate(self.disc_model):
            self.disc_optimizer.zero_grad()
            outputs = self._model_transform(self.model(inputs)).detach()
            disc_pred_real = self._disc_model_transform(
                self.disc_model(targets)
            )
            disc_pred_fake = self._disc_model_transform(
                self.disc_model(outputs)
            )

            disc_loss = self.disc_loss_fn(
                outputs,
                targets,
                disc_real=disc_pred_real,
                disc_fake=disc_pred_fake,
            )
            self.accelerator.backward(disc_loss)

            if self.accelerator.sync_gradients:
                if self.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        parameters=self.disc_model.parameters(),
                        max_norm=self.clip_grad_norm,
                    )
                if self.clip_grad_value:
                    self.accelerator.clip_grad_value_(
                        parameters=self.disc_model.parameters(),
                        clip_value=self.clip_grad_value,
                    )

            self.disc_optimizer.step()
            self.disc_scheduler.step()

        # Update generator
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            outputs = self._model_transform(self.model(inputs))
            disc_pred_fake = self._disc_model_transform(
                self.disc_model(outputs)
            )

            loss = self.loss_fn(
                outputs,
                targets,
                disc_fake=disc_pred_fake,
            )
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.clip_grad_norm,
                    )
                if self.clip_grad_value:
                    self.accelerator.clip_grad_value_(
                        parameters=self.model.parameters(),
                        clip_value=self.clip_grad_value,
                    )

            self.optimizer.step()
            self.scheduler.step()

        return self._output_transform(
            inputs, targets, outputs, loss, disc_loss
        )
