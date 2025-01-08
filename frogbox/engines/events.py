from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum


class EventStep(str, Enum):
    STARTED = "started"
    EPOCH_STARTED = "epoch_started"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EPOCH_COMPLETED = "epoch_completed"
    COMPLETED = "completed"


class MatchableEvent(ABC):
    @abstractmethod
    def matches(self, event: EventStep, step: int) -> bool: ...


class EventList(MatchableEvent):
    def __init__(self):
        self.events = []

    def __or__(self, other: MatchableEvent) -> "EventList":
        self.events.append(other)
        return self

    def matches(self, event: EventStep, step: int) -> bool:
        for entry in self.events:
            if entry.matches(event, step):
                return True
        return False


class Event(MatchableEvent):
    def __init__(
        self,
        event: str | EventStep,
        every: Optional[int] = None,
        first: Optional[int] = None,
        last: Optional[int] = None,
    ):
        self.event = EventStep(event)
        self.every = every
        self.first = first
        self.last = last

    def matches(self, event: EventStep, step: int) -> bool:
        if event != self.event:
            return False

        if self.first is not None and step < self.first:
            return False

        if self.last is not None and step > self.last:
            return False

        if self.every is not None and step % self.every != 0:
            return False

        return True

    def __or__(self, other: MatchableEvent) -> EventList:
        return EventList() | self | other
