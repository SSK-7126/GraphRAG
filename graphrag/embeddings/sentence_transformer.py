import numpy as np
from time import perf_counter
from graphrag.embeddings.embedder import Embedder

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *, device: str | None = None, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Embedding support requires the optional sentence-transformers dependency."
            ) from error
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.device = str(self.model.device)
        self.last_timing: dict[str, float | str | None] = {}

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_many(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """Batch encoding and synchronized CUDA timings when GPU is selected."""
        import torch
        effective_batch_size = batch_size or self.batch_size
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        # CUDA events measure queued GPU work. Host/device transfer is handled
        # inside SentenceTransformers and cannot be reliably separated here.
        if is_cuda:
            torch.cuda.synchronize()
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        wall_started = perf_counter()
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size or self.batch_size,
            show_progress_bar=False,
        )
        if is_cuda:
            gpu_end.record()
            torch.cuda.synchronize()
            gpu_seconds: float | None = gpu_start.elapsed_time(gpu_end) / 1000
        else:
            gpu_seconds = None
        self.last_timing = {
            "device": self.device,
            "items": len(texts),
            "batch_size": effective_batch_size,
            "wall_seconds": perf_counter() - wall_started,
            "gpu_compute_seconds": gpu_seconds,
            "host_device_transfer_seconds": None,
        }
        return embeddings
