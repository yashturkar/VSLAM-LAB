#!/usr/bin/env python3
"""Background worker for one timestamped Streamlit SLAM run."""

from __future__ import annotations

import argparse
import fcntl
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from Baselines.get_baseline import get_baseline, list_available_baselines
from Run.single_sequence import evaluate_single_trajectory, run_single_baseline
from Utilities.lightning_fastlio_pipeline import Pipeline, PipelinePaths, atomic_json, parse_topic_counts


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--fastlio", action="store_true")
    args = parser.parse_args()
    state_path = args.run_dir / "run.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    def update(**values: object) -> None:
        state.update(values, updated_at=now())
        atomic_json(state_path, state)

    try:
        if args.baseline not in list_available_baselines():
            raise ValueError(f"Unknown baseline: {args.baseline}")
        baseline = get_baseline(args.baseline)
        if "stereo" not in getattr(baseline, "modes", []):
            raise ValueError(f"Baseline does not support stereo: {args.baseline}")
        installed, reason = baseline.is_installed()
        if not installed:
            raise RuntimeError(f"Baseline is unavailable: {args.baseline} ({reason})")
        paths = PipelinePaths(args.sequence, args.local_root)
        pipeline = Pipeline(paths, args.workspace)
        manifest = json.loads((args.sequence / "manifest.json").read_text(encoding="utf-8"))
        processing = json.loads((args.sequence / "processing.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or processing.get("status") != "complete":
            raise RuntimeError("Capture and processing manifests must both be complete")
        counts = parse_topic_counts(args.sequence / "research-bag/metadata.yaml")
        update(status="preparing", stage="stage")
        paths.run_root.mkdir(parents=True, exist_ok=True)
        with paths.lock.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            stage_fp = pipeline.run_staging(pipeline.source_identity(), 1, 2)
            update(stage="extract")
            pipeline.run_extraction(stage_fp, counts, 2, 2)

        output = args.run_dir / "output"
        config = args.run_dir / "config.yaml"
        content = {
            "DATASET": {
                "base_path": str(paths.dataset_root),
                "name": paths.sequence_name,
                "baseline": args.baseline,
                "dataset": "lightning",
                "sensor_type": "stereo",
                "groundtruth_format": "kitti",
                "output_dir": str(output),
            },
            "EVALUATION": {
                "max_time_difference_s": 0.02,
                "fast_lio": {"enabled": args.fastlio, "workspace": str(args.workspace)},
            },
        }
        config.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
        update(status="running", stage="slam", prepared_sequence=str(paths.sequence), config=str(config))
        trajectory = run_single_baseline(config, headless=True)
        update(stage="evaluation", trajectory=str(trajectory))
        metrics = evaluate_single_trajectory(config, trajectory)
        update(status="complete", stage=None, finished_at=now(), metrics=str(metrics), report=str(output / "trajectory_report.pdf"))
        return 0
    except BaseException as error:
        update(status="failed", stage=None, finished_at=now(), error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
