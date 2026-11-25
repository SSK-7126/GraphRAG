from __future__ import annotations
from parallel.base import OptimizedSerialBackend

class CudaBackend(OptimizedSerialBackend):
    """CUDA accelerates embeddings, not string/LLM extraction or graph mutation."""
    name = "cuda"
    def cuda_info(self) -> dict:
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False, "reason": "torch.cuda.is_available() is false; CPU fallback selected"}
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True, "device": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda, "total_memory_bytes": props.total_memory,
                "recommended_embedding_batch_size": self.recommended_embedding_batch_size(props.total_memory),
            }
        except ImportError:
            return {"available": False, "reason": "PyTorch is not installed"}

    @staticmethod
    def recommended_embedding_batch_size(total_memory_bytes: int) -> int:
        """Conservative batch selection; users can override after benchmarking."""
        memory_gib = total_memory_bytes / 1024**3
        if memory_gib < 5:
            return 64
        if memory_gib < 9:
            return 128
        return 256
