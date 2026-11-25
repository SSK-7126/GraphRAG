"""Compare real benchmark summaries without inventing unavailable values."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--benchmarks", type=Path, default=Path("benchmarks")); parser.add_argument("--output", type=Path, default=Path("PERFORMANCE_COMPARISON.md")); args = parser.parse_args()
    summaries = []
    for path in sorted(args.benchmarks.glob("*/*/*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); payload["path"] = str(path); summaries.append(payload)
    serial_by_dataset = {
        Path(item["path"]).parent.name: item
        for item in summaries if item["backend"] == "serial"
    }
    lines = [
        "# Performance Comparison", "",
        "Only measured runs are shown; absent backends have no simulated result.",
        "The included fixture uses `MockLLM` and is a plumbing/scheduling check, not a production LLM throughput claim.", "",
        "| Backend | Run ID | Dataset | Wall-clock seconds | Speedup vs serial | Documents/s | Chunks/s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        wall = item["wall_clock_runtime_seconds"]
        dataset = Path(item["path"]).parent.name
        serial = serial_by_dataset.get(dataset)
        speedup = serial["wall_clock_runtime_seconds"] / wall if serial and wall else None
        speedup_text = f"{speedup:.3f}" if speedup is not None else "N/A"
        lines.append(f"| {item['backend']} | {item['run_id']} | {dataset} | {wall:.6f} | {speedup_text} | {item['documents_processed_per_second'] or 0:.4f} | {item['chunks_processed_per_second'] or 0:.4f} |")
    if not summaries: lines += ["", "No benchmark summaries were found."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
if __name__ == "__main__": main()
