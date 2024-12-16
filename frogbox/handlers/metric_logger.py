from typing import Mapping, Callable, Optional, Any
from torchmetrics import Metric
from ..engines.engine import Engine
from ..engines.events import EventStep


class MetricLogger:
    def __init__(
        self,
        metrics: Mapping[str, Metric],
        log_function: Callable[[Any], None],
        prefix: Optional[str] = None,
    ):
        self._metrics = metrics
        self._log_function = log_function
        if prefix is None:
            prefix = ""
        self._prefix = prefix

    def attach(self, engine: Engine) -> None:
        engine.add_output_handler(self._handle_output)
        engine.add_event_handler(
            event=EventStep.EPOCH_STARTED,
            function=self._epoch_started,
        )
        engine.add_event_handler(
            event=EventStep.EPOCH_COMPLETED,
            function=self._epoch_completed,
        )

    def _handle_output(self, outputs) -> None:
        for metric in self._metrics.values():
            metric(*outputs)

    def _epoch_started(self) -> None:
        for metric in self._metrics.values():
            metric.reset()

    def _epoch_completed(self) -> None:
        data = {}
        for label, metric in self._metrics.items():
            data[self._prefix + label] = metric.compute().item()
        self._log_function(data)
