from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from accelerate import Accelerator
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from .engine import Evaluator, Trainer


def _default_forward(model: Callable, *args) -> tuple[Any, ...]:
    return (model(args[0]),) + args[1:]


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
        schedulers: Mapping[str, LRScheduler],
        loss_fn: Callable[..., Tensor],
        forward: Callable[[Callable, Sequence[Any]], Sequence[Any]] | None = None,
        clip_grad_norm: float | None = None,
        clip_grad_value: float | None = None,
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
        batch: Sequence[Any],
    ):
        self.model.train()

        with accelerator.accumulate(self.model):
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            outputs = self.forward(self.model, *batch)

            loss = self.loss_fn(*outputs)
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
        forward: Callable[[Callable, Sequence[Any]], Sequence[Any]] | None = None,
    ):
        if forward is None:
            forward = _default_forward

        self.model = model
        self.forward = forward

        super().__init__(process_fn=self.process)

    def process(
        self,
        accelerator: Accelerator,
        batch: Sequence[Any],
    ):
        self.model.eval()

        with torch.no_grad():
            outputs = self.forward(self.model, *batch)

        outputs = accelerator.gather_for_metrics(outputs)
        return outputs
