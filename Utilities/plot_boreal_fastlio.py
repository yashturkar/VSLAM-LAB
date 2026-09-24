#!/usr/bin/env python3
"""Plot a Boreal FAST-LIO trajectory against its GNSS position trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(trajectory: Path, gnss: Path, output: Path) -> tuple[Path, Path]:
    poses = pd.read_csv(trajectory)
    fixes = pd.read_csv(gnss)
    pose_time = poses["ts (ns)"].to_numpy(dtype=np.int64)
    xy = poses[["tx (m)", "ty (m)"]].to_numpy(dtype=float)
    fix_time = fixes["timestamp"].to_numpy(dtype=np.int64)
    lat = np.deg2rad(fixes["latitude"].to_numpy(dtype=float))
    lon = np.deg2rad(fixes["longitude"].to_numpy(dtype=float))
    en = np.column_stack(((lon - lon[0]) * 6_371_000 * np.cos(lat[0]),
                          (lat - lat[0]) * 6_371_000))

    overlap = (pose_time >= fix_time[0]) & (pose_time <= fix_time[-1])
    matched = np.column_stack([
        np.interp((pose_time[overlap] - fix_time[0]) / 1e9,
                  (fix_time - fix_time[0]) / 1e9, en[:, axis])
        for axis in range(2)
    ])
    if len(matched) < 3:
        raise ValueError("Trajectory and GNSS need at least three overlapping samples")
    centered_pose = xy[overlap] - xy[overlap].mean(axis=0)
    centered_gnss = matched - matched.mean(axis=0)
    u, _, vt = np.linalg.svd(centered_pose.T @ centered_gnss)
    rotation = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
    aligned = (xy - xy[overlap].mean(axis=0)) @ rotation + matched.mean(axis=0)
    errors = np.linalg.norm(aligned[overlap] - matched, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    fig, (route, residual) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.2, 1]},
    )
    route.plot(en[:, 0], en[:, 1], "k.-", markersize=2.5, linewidth=1.3, label="GNSS (1 Hz)")
    route.plot(aligned[:, 0], aligned[:, 1], color="#e87022", linewidth=1.8,
               label="FAST-LIO (aligned)")
    route.scatter(aligned[0, 0], aligned[0, 1], color="#17813a", s=55, zorder=5, label="start")
    route.set(xlabel="East (m)", ylabel="North (m)", title="Route")
    route.axis("equal")
    route.grid(alpha=0.3)
    route.legend(loc="lower left")

    elapsed = (pose_time[overlap] - pose_time[overlap][0]) / 1e9
    residual.plot(elapsed, errors, color="#3d65a7", linewidth=1.5)
    residual.axhline(rmse, color="#e87022", linestyle="--", linewidth=1.2,
                     label=f"2D aligned RMSE: {rmse:.2f} m")
    residual.set(xlabel="Elapsed time (s)", ylabel="Horizontal difference (m)",
                 title="FAST-LIO versus GNSS")
    residual.grid(alpha=0.3)
    residual.legend()
    fig.suptitle(f"BorealHDR: {gnss.parent.parent.name}")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    png = output / "boreal_fastlio_vs_gnss.png"
    pdf = output / "boreal_fastlio_vs_gnss.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("gnss", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in plot(args.trajectory, args.gnss, args.output):
        print(path)


if __name__ == "__main__":
    main()
