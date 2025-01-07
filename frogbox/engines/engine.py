from typing import (
    Callable,
    Iterable,
    List,
    Any,
    Union,
    Dict,
    Mapping,
    Optional,
    Sequence,
)
import tqdm
from .events import EventStep, MatchableEvent, Event


class Engine:
    def __init__(
        self,
        process_fn: Callable,
        show_progress: bool = True,
        progress_label: Optional[str] = None,
    ):
        self.process_fn = process_fn
        self.show_progress = show_progress
        self.progress_label = progress_label

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

    def _get_progress_label(self) -> str:
        label = ""
        if self.max_epochs > 1:
            label += f"[{self.epoch+1}/{self.max_epochs}]"
        if self.progress_label is not None and len(self.progress_label) > 0:
            label = self.progress_label + " " + label
        return label

    def _get_data_iterator(
        self,
        loader: Iterable,
    ) -> Iterable:
        iterator = loader
        if self.show_progress:
            desc = self._get_progress_label()
            iterator = tqdm.tqdm(
                iterator,
                desc=desc,
                ncols=80,
                leave=False,
            )
        return iterator

    def _is_done(self) -> bool:
        return self.epoch >= self.max_epochs

    def add_event_handler(
        self,
        event: Union[str, EventStep, MatchableEvent],
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

    def run(self, loader: Iterable, max_epochs: int = 1) -> None:
        self.max_epochs = max_epochs

        if self._is_done():
            self.epoch = 0
            self.iteration = 0

        self._fire_event(EventStep.STARTED)

        while not self._is_done():
            self._fire_event(EventStep.EPOCH_STARTED)

            iterations = self._get_data_iterator(loader)
            for batch in iterations:
                self._fire_event(EventStep.ITERATION_STARTED)

                output = self.process_fn(batch)
                self._handle_output(output)

                self.iteration += 1
                self._fire_event(EventStep.ITERATION_COMPLETED)

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