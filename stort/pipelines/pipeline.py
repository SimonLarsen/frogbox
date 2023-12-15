from typing import Callable, Union
from abc import ABC, abstractmethod
from ignite.engine import Engine, Events, CallableEventWithFilter
import torch
from ..config import Config


class Pipeline(ABC):
    config: Config
    trainer: Engine
    device: torch.device

    @abstractmethod
    def run(self) -> None:
        ...

    def install_callback(
        self,
        event: Union[Events, CallableEventWithFilter],
        callback: Callable[["Pipeline"], None],
    ) -> None:
        self.trainer.add_event_handler(
            event_name=event,
            handler=callback,
            pipeline=self,
        )
