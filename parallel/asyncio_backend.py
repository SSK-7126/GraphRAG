from __future__ import annotations
import asyncio
from parallel.base import ExecutionBackend, ExtractionTask, ExtractionResult, execute_extraction_task, llm_spec

class AsyncioBackend(ExecutionBackend):
    name = "asyncio"
    def extract_chunks(self, llm, tasks: list[ExtractionTask], retry_count: int) -> list[ExtractionResult]:
        spec = llm_spec(llm); limit = min(self.workers, self.max_concurrent_llm_calls)
        async def run() -> list[ExtractionResult]:
            semaphore = asyncio.Semaphore(limit)
            async def one(index, task):
                async with semaphore:
                    return await asyncio.to_thread(execute_extraction_task, spec, task, retry_count, f"async-{index % limit}")
            return await asyncio.gather(*(one(index, task) for index, task in enumerate(tasks)))
        self.metrics.update(tasks=len(tasks), worker_count=limit)
        return asyncio.run(run())
