from abc import ABC, abstractmethod
from typing import Any


class Pipeline(ABC):
    @abstractmethod
    def run(self, input: str) -> str:
        pass
