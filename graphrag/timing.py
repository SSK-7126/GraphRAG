"""Run-isolated timing and resource records for GraphRAG experiments."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Iterator
import uuid

TIMING_COLUMNS = [
    "run_id", "backend", "dataset", "document_id", "chunk_id", "stage", "worker_id",
    "start_timestamp", "end_timestamp", "duration_seconds", "items_processed",
    "success_count", "failure_count",
]

@dataclass
class TimingRecord:
    run_id: str; backend: str; dataset: str; document_id: str | None; chunk_id: str | None
    stage: str; worker_id: str | None; start_timestamp: str; end_timestamp: str
    duration_seconds: float; items_processed: int; success_count: int; failure_count: int

class TimingRecorder:
    """Records wall clock separately from the sum of overlapping task time."""
    def __init__(self, run_id: str | None = None, *, backend: str = "serial"):
        self.run_id, self.backend = run_id or uuid.uuid4().hex, backend
        self.records: list[TimingRecord] = []
        self.llm_calls = self.failed_llm_calls = self.retry_count = 0
        self.input_tokens: int | None = 0; self.output_tokens: int | None = 0
        self._started = perf_counter(); self.parallel_metrics: dict[str, object] = {}

    @contextmanager
    def measure(self, dataset: str, stage: str, *, document_id: str | None = None,
                chunk_id: str | None = None, worker_id: str | None = None,
                items_processed: int = 1) -> Iterator[dict[str, int]]:
        started_at, started = datetime.now(timezone.utc), perf_counter()
        outcome = {"items_processed": items_processed, "success_count": items_processed, "failure_count": 0}
        try:
            yield outcome
        except Exception:
            outcome["success_count"], outcome["failure_count"] = 0, items_processed
            raise
        finally:
            self._append(dataset, stage, started_at, datetime.now(timezone.utc), perf_counter() - started,
                         document_id, chunk_id, worker_id, outcome)

    def record_completed_task(self, dataset: str, stage: str, *, duration_seconds: float,
                              document_id: str | None, chunk_id: str | None, worker_id: str | None,
                              items_processed: int, success_count: int | None = None,
                              failure_count: int = 0) -> None:
        now = datetime.now(timezone.utc)
        self._append(dataset, stage, now, now, duration_seconds, document_id, chunk_id, worker_id, {
            "items_processed": items_processed,
            "success_count": items_processed if success_count is None else success_count,
            "failure_count": failure_count,
        })

    def _append(self, dataset, stage, start_at, end_at, duration, document_id, chunk_id, worker_id, outcome) -> None:
        self.records.append(TimingRecord(self.run_id, self.backend, dataset, document_id, chunk_id, stage,
            worker_id, start_at.isoformat(), end_at.isoformat(), duration, outcome["items_processed"],
            outcome["success_count"], outcome["failure_count"]))

    def record_llm_usage(self, usage: dict | None, *, failed: bool = False, retry: bool = False) -> None:
        self.llm_calls += 1; self.failed_llm_calls += int(failed); self.retry_count += int(retry)
        if not usage:
            self.input_tokens = self.output_tokens = None; return
        for attribute, key in (("input_tokens", "prompt_eval_count"), ("output_tokens", "eval_count")):
            value = usage.get(key)
            if value is None: setattr(self, attribute, None)
            elif getattr(self, attribute) is not None: setattr(self, attribute, getattr(self, attribute) + int(value))

    def set_parallel_metrics(self, **metrics: object) -> None:
        self.parallel_metrics.update(metrics)

    def write(self, timings_dir: Path, *, write_summary: bool = True) -> dict:
        """Write this run only. Existing runs are represented by separate directories."""
        timings_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "stage_timings.csv": self.records,
            "document_timings.csv": [r for r in self.records if r.document_id],
            "chunk_timings.csv": [r for r in self.records if r.chunk_id],
            "query_timings.csv": [r for r in self.records if r.stage.startswith("query_")],
            "evaluation_timings.csv": [r for r in self.records if r.stage.startswith("evaluation_")],
        }
        for filename, records in files.items(): self._write_csv(timings_dir / filename, records)
        summary = self.summary()
        if write_summary:
            temporary = timings_dir / "summary.json.tmp"; temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temporary.replace(timings_dir / "summary.json")
        return summary

    @staticmethod
    def _write_csv(path: Path, records: list[TimingRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS); writer.writeheader()
            writer.writerows(asdict(record) for record in records)

    def summary(self) -> dict:
        wall = perf_counter() - self._started; aggregate = sum(record.duration_seconds for record in self.records)
        stages: dict[str, list[float]] = {}; documents, chunks = set(), set()
        for record in self.records:
            stages.setdefault(record.stage, []).append(record.duration_seconds)
            if record.document_id: documents.add(record.document_id)
            if record.chunk_id: chunks.add(record.chunk_id)
        return {
            "run_id": self.run_id, "backend": self.backend, "total_runtime_seconds": wall,
            "wall_clock_runtime_seconds": wall, "aggregate_stage_runtime_seconds": aggregate,
            "stages": {stage: {"aggregate_seconds": sum(values), "average_seconds": sum(values)/len(values),
                "median_seconds": median(values), "minimum_seconds": min(values), "maximum_seconds": max(values),
                "calls": len(values), "percent_of_wall_clock": sum(values)/wall*100 if wall else 0}
                for stage, values in sorted(stages.items())},
            "documents_processed_per_second": len(documents)/wall if wall else None,
            "chunks_processed_per_second": len(chunks)/wall if wall else None,
            "documents_seen": len(documents), "chunks_seen": len(chunks),
            "total_llm_calls": self.llm_calls, "failed_llm_calls": self.failed_llm_calls,
            "retry_count": self.retry_count, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "parallel_metrics": self.parallel_metrics,
        }
