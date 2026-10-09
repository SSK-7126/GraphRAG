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
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {details}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                "Cannot connect to Ollama. Check that Ollama is running."
            ) from error

        return data["message"]["content"]