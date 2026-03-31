from typing import Callable, Any
from collections.abc import Sequence, Mapping
from os import PathLike
import torch
from .pipeline import Pipeline
from ..config import SupervisedConfig, create_object_from_config
from ..engines.engine import Trainer, Evaluator
from ..engines.supervised import SupervisedTrainer, SupervisedEvaluator


class SupervisedPipeline(Pipeline):
    """Supervised pipeline."""

    config: SupervisedConfig
    trainer: SupervisedTrainer
    evaluator: SupervisedEvaluator

    def __init__(
        self,
        config: SupervisedConfig,
        checkpoint: str | PathLike | None = None,
        checkpoint_keys: Sequence[str] | None = None,
    ):
        """
        Create supervised pipeline.

        Parameters
        ----------
        config : SupervisedConfig
            Pipeline configuration.
        checkpoint : path-like
            Path to experiment checkpoint.
        checkpoint_keys : list of str
            List of keys for objects to load from checkpoint.
            Defaults to all keys.
        """

        def trainer_factory(
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, Mapping[str, torch.optim.Optimizer]],
            schedulers: Mapping[
                str, Mapping[str, torch.optim.lr_scheduler.LRScheduler]
            ],
            losses: Mapping[str, Callable[[Any, Any], Any]],
        ) -> Trainer:
            return SupervisedTrainer(
                model=models["model"],
                optimizers=optimizers["model"],
                schedulers=schedulers["model"],
                loss_fn=losses["model"],
                forward=(
                    create_object_from_config(config.trainer_forward)
                    if config.trainer_forward is not None
                    else None
                ),
                clip_grad_norm=config.clip_grad_norm,
                clip_grad_value=config.clip_grad_value,
            )

        def evaluator_factory(
            models: Mapping[str, torch.nn.Module]
        ) -> Evaluator:
            return SupervisedEvaluator(
                model=models["model"],
                forward=(
                    create_object_from_config(config.evaluator_forward)
                    if config.evaluator_forward is not None
                    else None
                ),
            )

        super().__init__(
            config=config,
            models={"model": config.model},
            losses={"model": config.losses},
            trainer=trainer_factory,
            evaluator=evaluator_factory,
            checkpoint=checkpoint,
            checkpoint_keys=checkpoint_keys,
        )

    def _get_checkpoint_dict(self) -> tuple[Mapping[str, Any], Sequence[str]]:
        to_save: dict[str, Any] = {
            "trainer": self.trainer,
            "model": self._models["model"],
        }
        to_unwrap = ["model"]
        for name, optimizer in self._optimizers["model"].items():
            to_save[f"optimizer_{name}"] = optimizer
        for name, scheduler in self._schedulers["model"].items():
            to_save[f"scheduler_{name}"] = scheduler
        return to_save, to_unwrap

    @property
    def model(self) -> torch.nn.Module:
        return self._models["model"]

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return self._optimizers["model"]
