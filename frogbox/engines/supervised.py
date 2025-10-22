from typing import Callable, Any, Optional, Tuple, Mapping
import torch
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator
from .engine import Trainer, Evaluator


def _default_forward(x: Any, y: Any, model: Callable) -> Tuple[Any, Any]:
    return y, model(x)


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
        schedulers: Mapping[str, LRScheduler],
        loss_fn: Callable[[Any, Any], Any],
        forward: Optional[
            Callable[[Any, Any, Callable], Tuple[Any, Any]]
        ] = None,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if forward is None:
            forward = _default_forward

        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.loss_fn = loss_fn
        self.forward = forward

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        super().__init__(process_fn=self.process)

    def process(
        self,
        accelerator: Accelerator,
        batch: Tuple[Any, Any],
    ):
        self.model.train()

        x, y = batch

        with accelerator.accumulate(self.model):
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            y, y_pred = self.forward(x, y, self.model)

            loss = self.loss_fn(y_pred, y)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if self.clip_grad_norm:
                    accelerator.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.clip_grad_norm,
                    )
                if self.clip_grad_value:
                    accelerator.clip_grad_value_(
                        parameters=self.model.parameters(),
                        clip_value=self.clip_grad_value,
                    )

            for optimizer in self.optimizers.values():
                optimizer.step()

            for scheduler in self.schedulers.values():
                scheduler.step()

        return loss.item()


class SupervisedEvaluator(Evaluator):
    def __init__(
        self,
        model: torch.nn.Module,
        forward: Optional[
            Callable[[Any, Any, Callable], Tuple[Any, Any]]
        ] = None,
    ):
        if forward is None:
            forward = _default_forward

        self.model = model
        self.forward = forward

        super().__init__(process_fn=self.process)

    def process(
        self,
        accelerator: Accelerator,
        batch: Tuple[Any, Any],
    ):
        self.model.eval()

        x, y = batch

        with torch.no_grad():
            y, y_pred = self.forward(x, y, self.model)

        y_pred, y = accelerator.gather_for_metrics((y_pred, y))
        return y_pred, y
