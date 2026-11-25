from __future__ import annotations
from parallel.base import BackendUnavailableError, ExecutionBackend, ExtractionTask, ExtractionResult, execute_extraction_task, llm_spec

class RayBackend(ExecutionBackend):
    name = "ray"
    def extract_chunks(self, llm, tasks: list[ExtractionTask], retry_count: int) -> list[ExtractionResult]:
        try: import ray
        except ImportError as error: raise BackendUnavailableError("Ray backend requires `pip install .[ray]`.") from error
        spec = llm_spec(llm); self.metrics.update(tasks=len(tasks), worker_count=self.workers)
        started_here = False
        if not ray.is_initialized(): ray.init(num_cpus=self.workers, ignore_reinit_error=True); started_here = True
        remote = ray.remote(execute_extraction_task)
        try:
            refs = [remote.remote(spec, task, retry_count, f"ray-{index % self.workers}") for index, task in enumerate(tasks)]
            return ray.get(refs)
        finally:
            if started_here: ray.shutdown()
