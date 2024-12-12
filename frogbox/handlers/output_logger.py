from typing import Callable, Any
from ..engines.engine import Engine


class OutputLogger:
    def __init__(
        self,
        label: str,
        log_function: Callable[[Any], None],
        output_transform: Callable[[Any], Any] = lambda x: x,
    ):
        self._label = label
        self._log_function = log_function
        self._output_transform = output_transform

    def attach(self, engine: Engine):
        engine.add_output_handler(self._handle_output)

    def _handle_output(self, output: Any) -> None:
        output = self._output_transform(output)
        self._log_function({self._label: output})
