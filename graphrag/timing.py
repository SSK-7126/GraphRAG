"""Runtime timing records for sequential GraphRAG runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Iterator
from contextlib import contextmanager
import uuid


TIMING_COLUMNS = [
    "run_id", "dataset", "document_id", "chunk_id", "stage", "start_timestamp",
    "end_timestamp", "duration_seconds", "items_processed", "success_count", "failure_count",
]


@dataclass
class TimingRecord:
    run_id: str
    dataset: str
    document_id: str | None
    chunk_id: str | None
    stage: str
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float
    items_processed: int
    success_count: int
    failure_count: int


class TimingRecorder:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self.records: list[TimingRecord] = []
        self.llm_calls = 0
        self.failed_llm_calls = 0
        self.retry_count = 0
        self.input_tokens: int | None = 0
        self.output_tokens: int | None = 0

    @contextmanager
    def measure(
        self, dataset: str, stage: str, *, document_id: str | None = None,
        chunk_id: str | None = None, items_processed: int = 1,
    ) -> Iterator[dict[str, int]]:
        start_at = datetime.now(timezone.utc)
        started = perf_counter()
        outcome = {
            "items_processed": items_processed,
            "success_count": items_processed,
            "failure_count": 0,
        }
        try:
            yield outcome
        except Exception:
            outcome["success_count"] = 0
            outcome["failure_count"] = items_processed
            raise
        finally:
            ended = perf_counter()
            end_at = datetime.now(timezone.utc)
            self.records.append(TimingRecord(
                self.run_id, dataset, document_id, chunk_id, stage,
                start_at.isoformat(), end_at.isoformat(), ended - started,
                outcome["items_processed"], outcome["success_count"], outcome["failure_count"],
            ))

    def record_llm_usage(self, usage: dict | None, *, failed: bool = False, retry: bool = False) -> None:
        self.llm_calls += 1
        self.failed_llm_calls += int(failed)
        self.retry_count += int(retry)
        if not usage:
            self.input_tokens = None
            self.output_tokens = None
            return
        prompt = usage.get("prompt_eval_count")
        completion = usage.get("eval_count")
        if prompt is None:
            self.input_tokens = None
        elif self.input_tokens is not None:
            self.input_tokens += int(prompt)
        if completion is None:
            self.output_tokens = None
        elif self.output_tokens is not None:
            self.output_tokens += int(completion)

    def write(self, timings_dir: Path, *, write_summary: bool = True) -> dict:
        timings_dir.mkdir(parents=True, exist_ok=True)
        by_file = {
            "stage_timings.csv": self.records,
            "document_timings.csv": [r for r in self.records if r.document_id],
            "query_timings.csv": [r for r in self.records if r.stage.startswith("query_")],
            "evaluation_timings.csv": [r for r in self.records if r.stage.startswith("evaluation_")],
        }
        for filename, records in by_file.items():
            self._write_csv(timings_dir / filename, records)
        summary = self.summary()
        if write_summary:
            temp = timings_dir / "summary.json.tmp"
            temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temp.replace(timings_dir / "summary.json")
        return summary

    @staticmethod
    def _write_csv(path: Path, records: list[TimingRecord]) -> None:
        has_rows = path.exists() and path.stat().st_size > 0
        with path.open("a" if has_rows else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS)
            if not has_rows:
                writer.writeheader()
            writer.writerows(asdict(record) for record in records)

    def summary(self) -> dict:
        total = sum(record.duration_seconds for record in self.records)
        stages: dict[str, list[float]] = {}
        documents = set()
        chunks = set()
        for record in self.records:
            stages.setdefault(record.stage, []).append(record.duration_seconds)
            if record.document_id:
                documents.add(record.document_id)
            if record.chunk_id:
                chunks.add(record.chunk_id)
        return {
            "run_id": self.run_id,
            "total_runtime_seconds": total,
            "stages": {
                stage: {
                    "total_seconds": sum(durations),
                    "percent_of_total": (sum(durations) / total * 100) if total else 0,
                    "average_seconds": sum(durations) / len(durations),
                    "median_seconds": median(durations),
                    "minimum_seconds": min(durations),
                    "maximum_seconds": max(durations),
                    "calls": len(durations),
                }
                for stage, durations in sorted(stages.items())
            },
            "documents_processed_per_second": len(documents) / total if total else None,
            "chunks_processed_per_second": len(chunks) / total if total else None,
            "documents_seen": len(documents),
            "chunks_seen": len(chunks),
            "total_llm_calls": self.llm_calls,
            "failed_llm_calls": self.failed_llm_calls,
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
