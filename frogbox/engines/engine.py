from typing import (
    Callable,
    Protocol,
    Iterator,
    List,
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
import tqdm
import torch
from accelerate import Accelerator
from .events import EventStep, MatchableEvent, Event
from ..pipelines.composite_loss import CompositeLoss


class SizedIterable(Protocol):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator:
        ...


class Engine:
    def __init__(
        self,
        process_fn: Callable[[Accelerator, Tuple[Any, Any]], Any],
    ):
        self.process_fn = process_fn

        self.epoch = 0
        self.iteration = 0
        self.max_epochs = 1

        self.event_handlers: List[EventHandler] = []
        self.output_handlers: List[OutputHandler] = []

    def _fire_event(self, event: EventStep) -> None:
        step = 0
        if event in (EventStep.EPOCH_STARTED, EventStep.EPOCH_COMPLETED):
            step = self.epoch
        if event in (
            EventStep.ITERATION_STARTED,
            EventStep.ITERATION_COMPLETED,
        ):
            step = self.iteration

        for handler in self.event_handlers:
            if handler.event.matches(event, step):
                handler.function(*handler.args, **handler.kwargs)

    def _handle_output(self, output: Any) -> None:
        for handler in self.output_handlers:
            handler.function(output)

    def _get_progress_label(self, prefix: Optional[str] = None) -> str:
        label = ""
        if self.max_epochs > 1:
            label += f"[{self.epoch+1}/{self.max_epochs}]"
        if prefix is not None and len(prefix) > 0:
            label = prefix + " " + label
        return label

    def _get_data_iterator(self, loader: SizedIterable) -> Iterator:
        return iter(loader)

    def _get_data_length(self, loader: SizedIterable) -> int:
        return len(loader)

    def _get_progress_bar(
        self,
        loader: SizedIterable,
        prefix: Optional[str] = None,
        disable: bool = False,
    ) -> tqdm.tqdm:
        desc = self._get_progress_label(prefix)
        return tqdm.tqdm(
            iterable=loader,
            desc=desc,
            ncols=80,
            leave=False,
            disable=disable,
        )

    def _is_done(self) -> bool:
        return self.epoch >= self.max_epochs

    def add_event_handler(
        self,
        event: str | EventStep | MatchableEvent,
        function: Callable[..., None],
        *args,
        **kwargs,
    ):
        if isinstance(event, str) or isinstance(event, EventStep):
            event = Event(event)
        self.event_handlers.append(
            EventHandler(event, function, *args, **kwargs)
        )

    def add_output_handler(
        self,
        function: Callable[[Any], None],
    ):
        self.output_handlers.append(OutputHandler(function))

    def run(
        self,
        accelerator: Accelerator,
        loader: SizedIterable,
        max_epochs: int = 1,
        show_progress: bool = True,
        progress_label: Optional[str] = None,
    ) -> None:
        self.max_epochs = max_epochs

        if self._is_done():
            self.epoch = 0
            self.iteration = 0

        self._fire_event(EventStep.STARTED)

        while not self._is_done():
            self._fire_event(EventStep.EPOCH_STARTED)

            iterations = self._get_data_iterator(loader)
            pbar = self._get_progress_bar(
                loader=loader,
                prefix=progress_label,
                disable=not show_progress,
            )
            epoch_length = self._get_data_length(loader)

            for _ in range(epoch_length):
                self._fire_event(EventStep.ITERATION_STARTED)

                batch = next(iterations)
                output = self.process_fn(accelerator, batch)
                self._handle_output(output)

                self.iteration += 1
                self._fire_event(EventStep.ITERATION_COMPLETED)
                pbar.update()

            pbar.close()

            self.epoch += 1
            self._fire_event(EventStep.EPOCH_COMPLETED)

        self._fire_event(EventStep.COMPLETED)

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            epoch=self.epoch,
            iteration=self.iteration,
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.epoch = state_dict["epoch"]
        self.iteration = state_dict["iteration"]


class Trainer(Engine):
    ...


class Evaluator(Engine):
    ...


class TrainerFactory(Protocol):
    def __call__(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, Mapping[str, torch.optim.Optimizer]],
        schedulers: Mapping[
            str, Mapping[str, torch.optim.lr_scheduler.LRScheduler]
        ],
        losses: Mapping[str, CompositeLoss],
    ) -> Trainer:
        ...


class EvaluatorFactory(Protocol):
    def __call__(
        self,
        models: Mapping[str, torch.nn.Module],
    ) -> Evaluator:
        ...


class EventHandler:
    def __init__(
        self,
        event: MatchableEvent,
        function: Callable[..., None],
        *args,
        **kwargs,
    ):
        self.event = event
        self.function = function
        self.args: Sequence[Any] = args
        self.kwargs: Dict[str, Any] = kwargs


class OutputHandler:
    def __init__(
        self,
        function: Callable[[Any], None],
    ):
        self.function = function
