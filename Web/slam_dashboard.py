"""Backend services shared by the Streamlit dashboard and worker."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Baselines.get_baseline import get_baseline, list_available_baselines
from Utilities.lightning_fastlio_pipeline import parse_topic_counts

DEFAULT_NAS_ROOT = Path("/mnt/share/nas/eph/clid-v2-sequences")
DEFAULT_LOCAL_ROOT = Path("/mnt/share/local/eph/VSLAM")
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    session: str
    path: str
    captured_at: str | None
    status: str
    stereo_frames: int
    odometry_poses: int
    lidar_scans: int
    size_gib: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _capture_time(name: str) -> str | None:
    match = re.search(r"_(20\d{6})_(\d{6})_\d+$", name)
    if not match:
        return None
    value = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return value.isoformat()


def discover_sequences(root: Path = DEFAULT_NAS_ROOT) -> list[SequenceRecord]:
    """Find complete processed CLID captures without writing to the NAS."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    records: list[SequenceRecord] = []
    for manifest_path in root.rglob("manifest.json"):
        capture = manifest_path.parent
        research = capture / "research-bag"
        metadata = research / "metadata.yaml"
        mcaps = list(research.glob("*.mcap")) if research.is_dir() else []
        if not metadata.is_file() or len(mcaps) != 1:
            continue
        manifest = _read_json(manifest_path)
        processing = _read_json(capture / "processing.json")
        counts = parse_topic_counts(metadata)
        status = "complete" if manifest.get("status") == processing.get("status") == "complete" else "incomplete"
        name = str(manifest.get("name") or capture.name)
        records.append(SequenceRecord(
            name=name,
            session=capture.parent.name,
            path=str(capture),
            captured_at=_capture_time(capture.name),
            status=status,
            stereo_frames=min(
                counts.get("/cam_sync/cam0/image_preview/compressed", 0),
                counts.get("/cam_sync/cam1/image_preview/compressed", 0),
            ),
            odometry_poses=counts.get("/odometry", 0),
            lidar_scans=counts.get("/ouster/points", 0),
            size_gib=round(mcaps[0].stat().st_size / 1024**3, 2),
        ))
    return sorted(records, key=lambda item: (item.captured_at or "", item.name), reverse=True)


def available_stereo_baselines() -> list[dict[str, Any]]:
    """Return registered stereo baselines and their runtime availability."""
    result: list[dict[str, Any]] = []
    for name in list_available_baselines():
        if name.endswith("-dev"):
            continue
        try:
            baseline = get_baseline(name)
            if "stereo" not in getattr(baseline, "modes", []):
                continue
            installed, reason = baseline.is_installed()
            result.append({"name": name, "installed": bool(installed), "reason": str(reason)})
        except Exception as error:
            result.append({"name": name, "installed": False, "reason": str(error)})
    return sorted(result, key=lambda row: (not row["installed"], row["name"]))


def results_root(local_root: Path = DEFAULT_LOCAL_ROOT) -> Path:
    return local_root.expanduser().resolve() / "web_results"


def create_run(
    sequence: Path,
    baseline: str,
    include_fastlio: bool,
    local_root: Path = DEFAULT_LOCAL_ROOT,
    workspace: Path = Path("~/humble_ws"),
) -> Path:
    """Create and launch a timestamped detached dashboard run."""
    sequence = sequence.expanduser().resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_root(local_root) / sequence.name / f"{timestamp}_{baseline}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": 1,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sequence": str(sequence),
        "baseline": baseline,
        "include_fastlio": include_fastlio,
        "run_dir": str(run_dir),
    }
    (run_dir / "run.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(REPO_ROOT / "Utilities/web_slam_worker.py"),
        "--sequence", str(sequence),
        "--baseline", baseline,
        "--run-dir", str(run_dir),
        "--local-root", str(local_root.expanduser().resolve()),
        "--workspace", str(workspace.expanduser().resolve()),
    ]
    if include_fastlio:
        command.append("--fastlio")
    log = (run_dir / "run.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    state = _read_json(run_dir / "run.json") or state
    if state.get("status") == "queued":
        state["status"] = "running"
    state.update(pid=process.pid, started_at=state.get("started_at") or datetime.now(timezone.utc).isoformat())
    (run_dir / "run.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return run_dir


def discover_runs(local_root: Path = DEFAULT_LOCAL_ROOT) -> list[dict[str, Any]]:
    root = results_root(local_root)
    if not root.is_dir():
        return []
    runs = []
    for state_path in root.glob("*/*/run.json"):
        state = _read_json(state_path)
        pid = state.get("pid")
        if state.get("status") == "running" and isinstance(pid, int):
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                state["status"] = "stale"
        state["state_path"] = str(state_path)
        runs.append(state)
    return sorted(runs, key=lambda row: row.get("created_at", ""), reverse=True)


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, comparison in metrics.get("comparisons", {}).items():
        ape = comparison.get("ape", {})
        association = comparison.get("association", {})
        rpe = comparison.get("rpe", {})
        rows.append({
            "comparison": name.replace("_", " "),
            "matched poses": association.get("matched_poses"),
            "translation APE RMSE (m)": ape.get("translation", {}).get("rmse"),
            "rotation APE RMSE (deg)": ape.get("rotation", {}).get("rmse"),
            "translation RPE RMSE @1m (m)": rpe.get("translation_rmse_1m", {}).get("rmse"),
            "rotation RPE RMSE @1m (deg)": rpe.get("rotation_rmse_1m", {}).get("rmse"),
        })
    return pd.DataFrame(rows)


def trajectory_frames(run: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load and align result trajectories into the robot-odometry frame."""
    run_dir = Path(run["run_dir"])
    output = run_dir / "output"
    metrics = _read_json(output / "metrics.json")
    sequence_path = Path(run.get("prepared_sequence", ""))
    paths = {
        "robot odometry": sequence_path / "groundtruth.csv",
        "vslam": output / "00000_KeyFrameTrajectory.csv",
        "fast lio": sequence_path / "references/fast_lio/trajectory.csv",
    }
    alignments = {
        "vslam": metrics.get("comparisons", {}).get("vslam_vs_robot_odometry", {}).get("alignment"),
        "fast lio": metrics.get("comparisons", {}).get("fast_lio_vs_robot_odometry", {}).get("alignment"),
    }
    result: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.is_file():
            continue
        frame = pd.read_csv(path).iloc[:, :4].apply(pd.to_numeric, errors="coerce").dropna()
        frame.columns = ["timestamp", "x", "y", "z"]
        alignment = alignments.get(name)
        if alignment:
            import numpy as np
            xyz = frame[["x", "y", "z"]].to_numpy()
            rotation = np.asarray(alignment["rotation"], dtype=float)
            translation = np.asarray(alignment["translation_m"], dtype=float)
            frame.loc[:, ["x", "y", "z"]] = (alignment.get("scale", 1.0) * (rotation @ xyz.T)).T + translation
        result[name] = frame
    return result
