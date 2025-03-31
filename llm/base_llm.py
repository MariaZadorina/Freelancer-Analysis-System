from abc import ABC
from abc import abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> str:
        pass
