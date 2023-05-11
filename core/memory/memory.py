from abc import ABC, abstractmethod
from typing import Any


class Memory(ABC):
    """The Memory class which stores some experiences."""

    def __init__(self):
        pass

    @abstractmethod
    def remember(self, experience: Any) -> Any:
        pass

    @abstractmethod
    def recall(self, clue: Any) -> Any:
        pass
