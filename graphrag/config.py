"""Configuration for the sequential, dataset-oriented GraphRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    """Return an expanded environment path, or a project-relative default."""
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path
    processed_dir: Path
    ollama_model: str
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    retry_count: int
    checkpoint_frequency: int
    enable_embeddings: bool
    embedding_model: str
    log_level: str

    @classmethod
    def from_environment(cls) -> "PipelineConfig":
        """Load settings without requiring a dotenv dependency.

        The defaults deliberately point at this repository's ``data`` folder.
        Set ``GRAPHRAG_RAW_DIR`` and ``GRAPHRAG_PROCESSED_DIR`` to use an
        external ``raw``/``processed`` pair.
        """
        config = cls(
            raw_dir=_path_from_env("GRAPHRAG_RAW_DIR", PROJECT_ROOT / "data" / "raw"),
            processed_dir=_path_from_env(
                "GRAPHRAG_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"
            ),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            chunk_size=_int_from_env("GRAPHRAG_CHUNK_SIZE", 500),
            chunk_overlap=_int_from_env("GRAPHRAG_CHUNK_OVERLAP", 50),
            retry_count=_int_from_env("GRAPHRAG_RETRY_COUNT", 2),
            checkpoint_frequency=_int_from_env("GRAPHRAG_CHECKPOINT_FREQUENCY", 25),
            enable_embeddings=os.getenv("GRAPHRAG_ENABLE_EMBEDDINGS", "false").lower()
            in {"1", "true", "yes"},
            embedding_model=os.getenv(
                "GRAPHRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            log_level=os.getenv("GRAPHRAG_LOG_LEVEL", "INFO"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("GRAPHRAG_CHUNK_SIZE must be greater than zero.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("GRAPHRAG_CHUNK_OVERLAP must be non-negative and smaller than chunk size.")
        if self.retry_count < 0:
            raise ValueError("GRAPHRAG_RETRY_COUNT must be non-negative.")
        if self.checkpoint_frequency <= 0:
            raise ValueError("GRAPHRAG_CHECKPOINT_FREQUENCY must be greater than zero.")
