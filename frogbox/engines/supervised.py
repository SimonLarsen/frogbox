from typing import Callable, Any, Optional
import torch
from accelerate import Accelerator
from .engine import Engine


class SupervisedTrainer(Engine):
    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        loss_fn: Callable[[Any, Any], Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
        model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[
            [Any, Any, Any, Any], Any
        ] = lambda x, y, y_pred, loss: loss.item(),
        **kwargs,
    ):
        self.accelerator = accelerator
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self._input_transform = input_transform
        self._model_transform = model_transform
        self._output_transform = output_transform

        super().__init__(process_fn=self.process, **kwargs)

    def process(self, batch):
        self.model.train()

        inputs, targets = batch
        inputs, targets = self._input_transform(inputs, targets)

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            outputs = self._model_transform(self.model(inputs))

            loss = self.loss_fn(outputs, targets)
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

        return self._output_transform(inputs, targets, outputs, loss)


class SupervisedEvaluator(Engine):
    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
        model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[
            [Any, Any, Any], Any
        ] = lambda x, y, y_pred: (y_pred, y),
        **kwargs,
    ):
        self.accelerator = accelerator
        self.model = model

        self._input_transform = input_transform
        self._model_transform = model_transform
        self._output_transform = output_transform

        super().__init__(process_fn=self.process, **kwargs)

    def process(self, batch):
        self.model.eval()

        inputs, targets = batch
        inputs, targets = self._input_transform(inputs, targets)

        with torch.no_grad():
            outputs = self._model_transform(self.model(inputs))

        outputs, targets = self._output_transform(inputs, targets, outputs)

        outputs, targets = self.accelerator.gather_for_metrics(
            (outputs, targets)
        )

        return outputs, targets
