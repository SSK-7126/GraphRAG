"""Discover and validate the extracted benchmark datasets without mutating them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


EXPECTED_DOCUMENT_COUNTS = {"hotpotqa": 5491, "kevin_scott": 1669, "msft": 41}
DATASET_NAMES = tuple(EXPECTED_DOCUMENT_COUNTS)


@dataclass(frozen=True)
class DiscoveredDocument:
    id: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class DiscoveredCsv:
    path: Path
    relative_path: str
    dataset: str
    kind: str
    columns: list[str]


@dataclass
class DatasetDiscovery:
    name: str
    documents: list[DiscoveredDocument] = field(default_factory=list)
    question_files: list[DiscoveredCsv] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_manifest(self, raw_dir: Path) -> dict:
        return {
            "dataset": self.name,
            "raw_dir": str(raw_dir),
            "documents": [
                {"id": item.id, "relative_path": item.relative_path}
                for item in self.documents
            ],
            "question_files": [
                {
                    "relative_path": item.relative_path,
                    "kind": item.kind,
                    "columns": item.columns,
                }
                for item in self.question_files
            ],
            "discovery_evidence": self.evidence,
            "warnings": self.warnings,
        }


class DatasetDiscoverer:
    """Classify known collections using file content *and* structure.

    Folder labels are useful evidence but are never the only signal.  This lets
    discovery flag an unexpected extraction rather than silently treating every
    directory called ``input`` as HotPotQA.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir.resolve()

    def discover(self) -> dict[str, DatasetDiscovery]:
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw dataset directory does not exist: {self.raw_dir}")
        result = {name: DatasetDiscovery(name) for name in DATASET_NAMES}
        paths_by_dataset: dict[str, list[Path]] = {name: [] for name in DATASET_NAMES}
        unclassified: list[Path] = []
        for path in sorted(self.raw_dir.rglob("*.txt")):
            dataset, evidence = self._classify_text(path)
            if dataset is None:
                unclassified.append(path)
                continue
            paths_by_dataset[dataset].append(path)
            if evidence not in result[dataset].evidence:
                result[dataset].evidence.append(evidence)

        seen_ids: dict[str, Path] = {}
        seen_filenames: dict[str, list[Path]] = {}
        for dataset, paths in paths_by_dataset.items():
            for path in paths:
                relative = path.relative_to(self.raw_dir).as_posix()
                document_id = self._document_id(dataset, relative)
                if document_id in seen_ids:
                    result[dataset].warnings.append(
                        f"Duplicate document ID {document_id}: {seen_ids[document_id]} and {path}"
                    )
                else:
                    seen_ids[document_id] = path
                seen_filenames.setdefault(path.name.casefold(), []).append(path)
                result[dataset].documents.append(
                    DiscoveredDocument(document_id, path, relative)
                )

        for filename, paths in seen_filenames.items():
            if len(paths) > 1:
                owners = {self._classify_text(path)[0] for path in paths}
                for owner in owners - {None}:
                    result[owner].warnings.append(
                        f"Filename collision for {filename}: " + ", ".join(map(str, paths))
                    )
        if unclassified:
            message = "Unclassified text files (not skipped): " + ", ".join(
                str(path.relative_to(self.raw_dir)) for path in unclassified
            )
            for discovery in result.values():
                discovery.warnings.append(message)

        for csv_path in sorted(self.raw_dir.rglob("*.csv")):
            discovered = self._classify_csv(csv_path)
            if discovered is None:
                for discovery in result.values():
                    discovery.warnings.append(f"Unclassified CSV (not ingested): {csv_path}")
            else:
                result[discovered.dataset].question_files.append(discovered)
        archives = sorted(self.raw_dir.rglob("*.zip"))
        if archives:
            message = (
                "Nested archive(s) detected but not extracted automatically: "
                + ", ".join(str(path.relative_to(self.raw_dir)) for path in archives)
            )
            for discovery in result.values():
                discovery.warnings.append(message)
        return result

    def validate(self, discoveries: dict[str, DatasetDiscovery]) -> dict[str, list[str]]:
        issues: dict[str, list[str]] = {}
        for name, discovery in discoveries.items():
            dataset_issues = list(discovery.warnings)
            expected = EXPECTED_DOCUMENT_COUNTS[name]
            if len(discovery.documents) != expected:
                dataset_issues.append(
                    f"Expected {expected} text documents but discovered {len(discovery.documents)}."
                )
            expected_csvs = 2 if name == "msft" else 1
            if len(discovery.question_files) != expected_csvs:
                dataset_issues.append(
                    f"Expected {expected_csvs} question CSV file(s) but discovered "
                    f"{len(discovery.question_files)}."
                )
            issues[name] = dataset_issues
        return issues

    def _classify_text(self, path: Path) -> tuple[str | None, str]:
        relative = path.relative_to(self.raw_dir).as_posix().casefold()
        filename = path.name.casefold()
        sample = path.read_text(encoding="utf-8", errors="replace")[:1000].casefold()
        if "behind the tech" in filename or "kevin scott" in relative:
            return "kevin_scott", "podcast transcript filename/content pattern"
        if filename.startswith("transcript") or "earnings call" in sample or "msft input text" in relative:
            return "msft", "earnings-transcript filename/content pattern"
        if path.parent.resolve() == self.raw_dir / "input" and filename.startswith("test_"):
            return "hotpotqa", "flat test_N document pattern plus encyclopedic source text"
        return None, "no recognized content pattern"

    def _classify_csv(self, path: Path) -> DiscoveredCsv | None:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
        stem = path.stem.casefold()
        relative = path.relative_to(self.raw_dir).as_posix()
        if "hotpot" in stem:
            dataset, kind = "hotpotqa", "benchmark"
        elif "kevin" in stem or "scott" in stem:
            dataset, kind = "kevin_scott", "benchmark"
        elif "msft" in stem and "single" in stem:
            dataset, kind = "msft", "single_transcript"
        elif "msft" in stem and "multi" in stem:
            dataset, kind = "msft", "multi_transcript"
        else:
            return None
        if not any(column.casefold() in {"question", "question_text"} for column in columns):
            return None
        return DiscoveredCsv(path, relative, dataset, kind, columns)

    @staticmethod
    def _document_id(dataset: str, relative_path: str) -> str:
        # Dataset qualification preserves duplicate basenames and makes IDs stable
        # across Windows and POSIX path separators.
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
        return f"{dataset}:{digest}"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
