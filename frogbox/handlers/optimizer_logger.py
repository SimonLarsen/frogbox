from typing import Sequence, Callable, Any
import torch
from ..engines.engine import Engine
from ..engines.events import EventStep


class OptimizerLogger:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Sequence[str],
        log_function: Callable[[Any], None],
        prefix: str = "optimizer/",
    ):
        self._optimizer = optimizer
        self._params = params
        self._log_function = log_function
        if prefix is None:
            prefix = ""
        self._prefix = prefix

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(
            event=EventStep.ITERATION_COMPLETED,
            function=self._iteration_completed,
        )

    def _iteration_completed(self) -> None:
        data = {}
        num_groups = len(self._optimizer.param_groups)
        for i, param_group in enumerate(self._optimizer.param_groups):
            for param in self._params:
                label = self._prefix + param
                if num_groups > 1:
                    label = label + f"/group_{i}"
                data[label] = param_group[param]
        self._log_function(data)
