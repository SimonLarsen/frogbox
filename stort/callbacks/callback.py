from typing import Union, Dict, Callable
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events, CallableEventWithFilter
from ..config import Config


@dataclass
class CallbackState:
    trainer: Engine
    evaluator: Engine
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]
    model: torch.nn.Module
    config: Config
    device: torch.device


class Callback:
    def __init__(
        self,
        event: Union[Events, CallableEventWithFilter],
        function: Callable[[CallbackState], None],
    ):
        self._event = event
        self._function = function

    @property
    def event(self) -> Union[Events, CallableEventWithFilter]:
        return self._event

    @property
    def function(self) -> Callable[[CallbackState], None]:
        return self._function
