from typing import Callable, Any, Optional, Tuple, Mapping
import torch
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator
from .engine import Trainer, Evaluator


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
        schedulers: Mapping[str, LRScheduler],
        loss_fn: Callable[[Any, Any], Any],
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.loss_fn = loss_fn

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        super().__init__(process_fn=self.process)

    def process(
        self,
        accelerator: Accelerator,
        batch: Tuple[Any, Any],
    ):
        self.model.train()

        inputs, targets = batch

        with accelerator.accumulate(self.model):
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)
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
    def __init__(self, model: torch.nn.Module):
        self.model = model

        super().__init__(process_fn=self.process)

    def process(
        self,
        accelerator: Accelerator,
        batch: Tuple[Any, Any],
    ):
        self.model.eval()

        inputs, targets = batch

        with torch.no_grad():
            outputs = self.model(inputs)

        outputs, targets = accelerator.gather_for_metrics(
            (outputs, targets)
        )

        return outputs, targets
