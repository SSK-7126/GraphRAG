from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from parallel.base import ExecutionBackend, ExtractionTask, ExtractionResult, execute_extraction_task, llm_spec

class MultiprocessingBackend(ExecutionBackend):
    name = "multiprocessing"
    def extract_chunks(self, llm, tasks: list[ExtractionTask], retry_count: int) -> list[ExtractionResult]:
        spec = llm_spec(llm); self.metrics.update(tasks=len(tasks), worker_count=self.workers)
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(execute_extraction_task, spec, task, retry_count, f"process-{index % self.workers}") for index, task in enumerate(tasks)]
            return [future.result() for future in futures]
