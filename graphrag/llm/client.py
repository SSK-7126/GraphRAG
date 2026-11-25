from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @property
    def last_usage(self) -> dict | None:
        """Provider-reported token usage for the most recent call, if available."""
        return None
