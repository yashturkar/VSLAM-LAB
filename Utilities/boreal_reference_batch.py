#!/usr/bin/env python3
"""Generate, validate, and publish Boreal FAST-LIO reference sidecars."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Utilities.boreal_reference import ensure_full_reference  # noqa: E402
from Utilities.boreal_shared_reference import (SHARED_ROOT, publish_reference,  # noqa: E402
                                                shared_trajectory, validate_trajectory)

DEFAULT_SOURCE = Path("/mnt/share/local/eph/BorealHDR")
DEFAULT_LOCAL = Path("/mnt/share/local/eph/VSLAM/borealhdr/fastlio_reference")
DEFAULT_REPORT = Path("/mnt/share/local/eph/VSLAM/borealhdr/reference_batch")


def write_report(folder: Path, results: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "report.json.tmp"
    temporary.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(),
                                     "results": results}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(folder / "report.json")
    lines = ["# Boreal FAST-LIO shared references", "",
             "FAST-LIO is an experimental lidar/IMU reference, not surveyed ground truth.", "",
             "| Sequence | Status | Poses | Sensor coverage | Camera coverage | GNSS 2D RMSE | Path / failure |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for result in results:
        quality = result.get("quality", {})
        coverage = f"{quality['temporal_coverage_ratio']:.1%}" if quality else "—"
        camera = (f"{quality['camera_reference_pairs']}/{quality['camera_pairs']}"
                  if "camera_pairs" in quality else "—")
        rmse = (f"{quality['gnss_aligned_2d_rmse_m']:.2f} m"
                if "gnss_aligned_2d_rmse_m" in quality else "—")
        lines.append(f"| {result['sequence']} | {result['status']} | {quality.get('pose_count', '—')} "
                     f"| {coverage} | {camera} | {rmse} "
                     f"| {result.get('trajectory', result.get('error', '—'))} |")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Process every available backpack sequence")
    selection.add_argument("--sequence", action="append", help="Process one named sequence; repeatable")
    parser.add_argument("--check-only", action="store_true", help="Validate published references; do not generate or write to NAS")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-root", type=Path, default=SHARED_ROOT)
    parser.add_argument("--local-output", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    from Utilities.borealhdr import discover
    source_root = args.source_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    available = discover(source_root)
    names = available if args.all else list(dict.fromkeys(args.sequence))
    if not names:
        parser.error(f"No Boreal sequences found under {source_root}")
    unknown = sorted(set(names) - set(available))
    if unknown:
        parser.error("Unavailable Boreal sequences: " + ", ".join(unknown))
    if reference_root.is_relative_to(Path("/mnt/share/nas")) and not os.path.ismount("/mnt/share/nas"):
        parser.error("/mnt/share/nas is not mounted; refusing to publish to an unmounted directory")
    results: list[dict] = []
    for index, name in enumerate(names, 1):
        source = source_root / name
        print(f"[{index}/{len(names)}] {name}: checking", flush=True)
        try:
            shared = shared_trajectory(source, reference_root)
            if shared is not None:
                status, trajectory = "reused", shared
            elif (reference_root / name).exists():
                raise ValueError(f"Invalid shared bundle exists: {reference_root / name}")
            elif args.check_only:
                results.append({"sequence": name, "status": "missing"})
                print(f"[{index}/{len(names)}] {name}: missing", flush=True)
                write_report(args.report_root, results)
                continue
            else:
                local = ensure_full_reference(source, args.local_output, reference_root)
                trajectory = publish_reference(source, local, reference_root)
                if shared_trajectory(source, reference_root) != trajectory:
                    raise RuntimeError("Published bundle did not pass readback validation")
                status = "published"
            quality = validate_trajectory(source, trajectory)
            results.append({"sequence": name, "status": status,
                            "trajectory": str(trajectory), "quality": quality})
            print(f"[{index}/{len(names)}] {name}: {status}; {quality['pose_count']} poses; "
                  f"coverage {quality['temporal_coverage_ratio']:.1%}; "
                  f"camera {quality.get('camera_reference_pairs', '—')}/{quality.get('camera_pairs', '—')}; "
                  f"GNSS RMSE {quality.get('gnss_aligned_2d_rmse_m', float('nan')):.2f} m", flush=True)
        except Exception as error:
            results.append({"sequence": name, "status": "failed", "error": str(error)})
            print(f"[{index}/{len(names)}] {name}: FAILED: {error}", file=sys.stderr, flush=True)
        write_report(args.report_root, results)
    published = sum(result["status"] in {"published", "reused"} for result in results)
    print(f"Completed {published}/{len(results)} validated references; "
          f"report: {args.report_root / 'report.md'}", flush=True)
    if published != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
