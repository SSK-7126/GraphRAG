from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @property
    def last_usage(self) -> dict | None:
        """Provider-reported token usage for the most recent call, if available."""
        return None

    def worker_spec(self) -> dict:
        """Return a safe constructor specification for isolated worker clients."""
        raise TypeError(
            f"{type(self).__name__} cannot be used by a process/distributed backend; "
            "implement worker_spec() or use the serial backend."
        )
