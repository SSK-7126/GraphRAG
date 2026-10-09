"""Small OpenAI-compatible LLM client with no mandatory SDK dependency."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from graphrag.llm.client import LLM


class OpenAICompatibleLLM(LLM):
    """Call an OpenAI-compatible chat-completions endpoint when an API key is set."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY before creating OpenAICompatibleLLM.")

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(f"LLM request failed with HTTP {error.code}.") from error
        except URLError as error:
            raise RuntimeError("Could not reach the LLM endpoint.") from error
        return data["choices"][0]["message"]["content"]
