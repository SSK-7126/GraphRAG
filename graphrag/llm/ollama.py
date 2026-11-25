import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from graphrag.llm.client import LLM


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._last_usage: dict | None = None

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")

        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if error.code == 404:
                raise RuntimeError(
                    f"Ollama model {self.model!r} was not found. Run 'ollama pull {self.model}' "
                    "yourself, then retry."
                ) from error
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {details}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Check that its server is running."
            ) from error
        if "message" not in data or "content" not in data["message"]:
            raise RuntimeError("Ollama returned a response without message content.")
        self._last_usage = {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        }
        return data["message"]["content"]

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
