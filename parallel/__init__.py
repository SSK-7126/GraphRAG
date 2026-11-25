"""Selectable GraphRAG parallel execution backends."""
from parallel.base import ExecutionBackend, SerialBackend, OptimizedSerialBackend, BackendUnavailableError
from parallel.asyncio_backend import AsyncioBackend
from parallel.multiprocessing_backend import MultiprocessingBackend
from parallel.ray_backend import RayBackend
from parallel.dask_backend import DaskBackend
from parallel.cuda_backend import CudaBackend

BACKENDS = {cls.name: cls for cls in (SerialBackend, OptimizedSerialBackend, AsyncioBackend, MultiprocessingBackend, RayBackend, DaskBackend, CudaBackend)}
def create_backend(name: str, workers: int = 1, max_concurrent_llm_calls: int | None = None) -> ExecutionBackend:
    try: return BACKENDS[name](workers, max_concurrent_llm_calls=max_concurrent_llm_calls)
    except KeyError as error: raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}") from error
