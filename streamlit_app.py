"""VSLAM-LAB sequence browser and headless run dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from Web.slam_dashboard import (
    DEFAULT_LOCAL_ROOT,
    DEFAULT_NAS_ROOT,
    available_stereo_baselines,
    create_run,
    discover_runs,
    discover_sequences,
    metrics_table,
    trajectory_frames,
)

st.set_page_config(page_title="VSLAM-LAB", page_icon="🛰️", layout="wide")
st.title("VSLAM-LAB")
st.caption("Browse processed CLID captures, launch timestamped headless runs, and inspect EVO metrics.")

with st.sidebar:
    nas_root = Path(st.text_input("Sequence root", str(DEFAULT_NAS_ROOT)))
    local_root = Path(st.text_input("Local workspace", str(DEFAULT_LOCAL_ROOT)))
    if st.button("Refresh", width="stretch"):
        st.cache_data.clear()
    auto_refresh = st.toggle("Refresh running jobs every 5 seconds", value=True)


@st.cache_data(ttl=10)
def sequences(root: str):
    return discover_sequences(Path(root))


@st.cache_data(ttl=30)
def baselines():
    return available_stereo_baselines()


captures_tab, results_tab = st.tabs(["Captures", "Runs & results"])
with captures_tab:
    records = sequences(str(nas_root))
    if not records:
        st.warning(f"No processed captures found below {nas_root}")
    else:
        query = st.text_input("Filter captures", placeholder="session or sequence name").lower()
        filtered = [item for item in records if query in f"{item.session} {item.name}".lower()]
        st.dataframe(
            [{"session": row.session, "sequence": row.name, "captured": row.captured_at, "status": row.status,
              "stereo frames": row.stereo_frames, "odometry poses": row.odometry_poses,
              "LiDAR scans": row.lidar_scans, "MCAP GiB": row.size_gib} for row in filtered],
            width="stretch", hide_index=True,
        )
        selected_name = st.selectbox("Sequence", [f"{row.session} / {row.name}" for row in filtered])
        selected = filtered[[f"{row.session} / {row.name}" for row in filtered].index(selected_name)]
        compatible = baselines()
        runnable = [item["name"] for item in compatible if item["installed"]]
        unavailable = [f"{item['name']}: {item['reason']}" for item in compatible if not item["installed"]]
        left, right = st.columns(2)
        baseline = left.selectbox("SLAM", runnable, disabled=not runnable)
        fastlio = right.checkbox("Generate/use FAST-LIO reference", value=True)
        if unavailable:
            with st.expander("Unavailable stereo baselines"):
                st.write("\n".join(f"- {value}" for value in unavailable))
        if st.button("Run headlessly", type="primary", disabled=selected.status != "complete" or not runnable):
            run_dir = create_run(Path(selected.path), baseline, fastlio, local_root)
            st.success(f"Started {run_dir.name}")

with results_tab:
    runs = discover_runs(local_root)
    if not runs:
        st.info("No dashboard runs yet.")
    else:
        labels = [f"{row.get('created_at', '')} · {row.get('baseline')} · {Path(row.get('sequence', '')).name}" for row in runs]
        selected_run = runs[labels.index(st.selectbox("Run", labels))]
        status = selected_run.get("status", "unknown")
        st.subheader(status.upper())
        st.json({key: selected_run.get(key) for key in ("created_at", "started_at", "finished_at", "baseline", "sequence", "stage", "error") if selected_run.get(key) is not None})
        log_path = Path(selected_run["run_dir"]) / "run.log"
        if log_path.is_file():
            with st.expander("Run log", expanded=status in {"failed", "stale"}):
                st.code("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]))
        metrics_path = Path(selected_run["run_dir"]) / "output/metrics.json"
        if metrics_path.is_file():
            metric_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.subheader("EVO metrics")
            st.dataframe(metrics_table(metric_data), width="stretch", hide_index=True)
            frames = trajectory_frames(selected_run)
            if frames:
                xy, xz = st.columns(2)
                for container, axes, title in ((xy, ("x", "y"), "Top view"), (xz, ("x", "z"), "Side view")):
                    figure = go.Figure()
                    for name, frame in frames.items():
                        figure.add_trace(go.Scatter(x=frame[axes[0]], y=frame[axes[1]], mode="lines", name=name))
                    figure.update_layout(title=title, xaxis_title=f"{axes[0]} (m)", yaxis_title=f"{axes[1]} (m)", yaxis_scaleanchor="x")
                    container.plotly_chart(figure, width="stretch")
            report = Path(selected_run["run_dir"]) / "output/trajectory_report.pdf"
            c1, c2 = st.columns(2)
            c1.download_button("Download metrics.json", metrics_path.read_bytes(), "metrics.json", "application/json")
            if report.is_file():
                c2.download_button("Download trajectory PDF", report.read_bytes(), "trajectory_report.pdf", "application/pdf")

if auto_refresh and any(run.get("status") in {"queued", "running", "preparing"} for run in discover_runs(local_root)):
    st.markdown("<meta http-equiv='refresh' content='5'>", unsafe_allow_html=True)
