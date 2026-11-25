from __future__ import annotations
from parallel.base import BackendUnavailableError, ExecutionBackend, ExtractionTask, ExtractionResult, execute_extraction_task, llm_spec

class DaskBackend(ExecutionBackend):
    name = "dask"
    def extract_chunks(self, llm, tasks: list[ExtractionTask], retry_count: int) -> list[ExtractionResult]:
        try: from dask import compute, delayed
        except ImportError as error: raise BackendUnavailableError("Dask backend requires `pip install .[dask]`.") from error
        spec = llm_spec(llm); self.metrics.update(tasks=len(tasks), worker_count=self.workers)
        work = [delayed(execute_extraction_task)(spec, task, retry_count, f"dask-{index % self.workers}") for index, task in enumerate(tasks)]
        return list(compute(*work, scheduler="threads", num_workers=self.workers))
