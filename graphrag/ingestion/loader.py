from pathlib import Path
import re

def load_text_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    return path.read_text(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    """Perform deterministic, loss-minimising whitespace cleanup."""
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
