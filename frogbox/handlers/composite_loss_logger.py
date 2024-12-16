from typing import Optional, Callable, Any, Dict
from ..engines.events import EventStep
from ..engines.engine import Engine
from ..pipelines.composite_loss import CompositeLoss


class CompositeLossLogger:
    def __init__(
        self,
        loss: CompositeLoss,
        log_function: Callable[[Any], None],
        prefix: Optional[str] = None,
    ):
        self._loss = loss
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
        data: Dict[str, Any] = {}
        for label, loss in zip(self._loss.labels, self._loss.last_values):
            data[self._prefix + label] = loss
        self._log_function(data)
