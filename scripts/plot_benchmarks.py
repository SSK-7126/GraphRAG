"""Render benchmark charts from actual summary files only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib (declared as a project dependency).") from error
    rows = []
    for path in sorted(args.benchmarks.glob("*/*/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        # Discovery-only ('all') rows are not comparable to a dataset index run.
        if path.parent.name != "all": rows.append((data, path.parent.name))
    if not rows:
        raise SystemExit("No benchmark summaries found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{data['backend']}\n{dataset}" for data, dataset in rows]
    values = [data["wall_clock_runtime_seconds"] for data, _ in rows]
    plt.figure(figsize=(max(6, len(rows) * 1.3), 4))
    plt.bar(labels, values); plt.ylabel("Wall-clock seconds"); plt.title("Measured GraphRAG runtime")
    plt.tight_layout(); plt.savefig(args.output_dir / "total_runtime_by_backend.png", dpi=160); plt.close()
    stage_names = sorted({stage for data, _ in rows for stage in data["stages"]})
    if stage_names:
        width = 0.8 / len(rows); positions = list(range(len(stage_names)))
        plt.figure(figsize=(max(8, len(stage_names) * 0.7), 4.5))
        for index, (data, dataset) in enumerate(rows):
            values = [data["stages"].get(stage, {}).get("aggregate_seconds", 0) for stage in stage_names]
            plt.bar([position + index * width for position in positions], values, width, label=f"{data['backend']}:{dataset}")
        plt.xticks([position + width * (len(rows) - 1) / 2 for position in positions], stage_names, rotation=45, ha="right")
        plt.ylabel("Aggregate seconds"); plt.title("Measured stage time (overlap may exceed wall clock)"); plt.legend()
        plt.tight_layout(); plt.savefig(args.output_dir / "stage_runtime_by_backend.png", dpi=160); plt.close()

if __name__ == "__main__": main()
