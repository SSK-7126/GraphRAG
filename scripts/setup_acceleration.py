"""Cross-platform optional CUDA, Ray, and Dask bootstrap for GraphRAG."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def run(*arguments: str) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install optional GraphRAG acceleration packages.")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-ray", action="store_true")
    parser.add_argument("--skip-dask", action="store_true")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required; install it first or install the optional packages manually.")
    python = sys.executable
    if not args.skip_ray:
        run(uv, "pip", "install", "--python", python, "ray>=2.40")
    if not args.skip_dask:
        run(uv, "pip", "install", "--python", python, "dask>=2024.12")
    if args.skip_cuda or shutil.which("nvidia-smi") is None:
        print("CUDA skipped: no NVIDIA driver found or --skip-cuda was supplied.")
    else:
        # CUDA runtime libraries are bundled with the wheel; installing a CUDA
        # Toolkit separately is unnecessary for this Python-only project.
        run(
            uv, "pip", "install", "--python", python, "--upgrade",
            "--index-strategy", "unsafe-best-match", "--index-url",
            "https://download.pytorch.org/whl/cu128", "torch==2.11.0+cu128",
        )
        run(uv, "pip", "install", "--python", python, "sentence-transformers>=6.1.0")
    run(python, "run.py", "--hardware")


if __name__ == "__main__":
    main()
