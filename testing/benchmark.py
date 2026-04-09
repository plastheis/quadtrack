"""QuadTrack — offline tracker benchmark entry point.

Evaluates one or more trackers against an Anti-UAV dataset split and writes
per-sequence + aggregate statistics to a JSON file.

Usage:
    python testing/benchmark.py [config.yaml]

All behaviour is controlled by the ``benchmark:`` section of config.yaml.
See config.yaml for available options.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Allow running from repo root or from within testing/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.datasets.anti_uav import AntiUAVDataset
from benchmark.runner import BenchmarkRunner
from trackers.factory import build_from_benchmark_section
from benchmark.writer import print_aggregate_summary, write_json


def main(config_path: str = "config.yaml") -> None:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bm = cfg["benchmark"]

    # Resolve relative paths against the config file's directory (repo root),
    # not the shell's current working directory.
    config_dir = config_path.parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else config_dir / path

    # --- Dataset ---
    dataset = AntiUAVDataset(
        root=_resolve(bm["dataset_root"]),
        split=bm["split"],
        modality=bm.get("modality", "IR"),
    )

    # --- Trackers + Fusion ---
    trackers, fusion = build_from_benchmark_section(cfg)

    # --- Runner ---
    runner = BenchmarkRunner(dataset, trackers, fusion, cfg)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stats = runner.run()

    # --- Output ---
    output_dir = _resolve(bm.get("output_dir", "benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_json(stats, output_dir, timestamp)

    print_aggregate_summary(stats, out_path)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
