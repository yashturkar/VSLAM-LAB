"""Portable single-sequence execution shared by headless evaluation and GUI demo."""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from Baselines.get_baseline import get_baseline, list_available_baselines
from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.get_dataset import get_dataset, list_available_datasets
from Evaluate.pairwise_metrics import write_combined_report, write_pairwise_metrics
from Run.fastlio_reference import fast_lio_reference_enabled, generate_fast_lio_reference
from Run.run_functions import run_sequence
from path_constants import TRAJECTORY_FILE_NAME, VSLAM_LAB_DIR
from utilities import load_yaml_file, print_msg, ws

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


@dataclass(frozen=True)
class SingleSequenceConfig:
    base_path: Path
    name: str
    baseline: str
    dataset: str
    output_dir: Path
    sensor_type: str = "mono"
    groundtruth_format: str = "kitti"
    parameters: dict[str, Any] = field(default_factory=dict)
    max_time_difference_s: float = 0.02

    @classmethod
    def load(cls, config_yaml: str | Path) -> "SingleSequenceConfig":
        config_path = Path(config_yaml).expanduser().resolve()
        data = load_yaml_file(config_path)
        section = data.get("DATASET") if isinstance(data, dict) else None
        if not isinstance(section, dict):
            raise ValueError("Single-sequence config must contain a DATASET mapping")
        missing = [key for key in ("base_path", "name", "baseline", "dataset") if not section.get(key)]
        if missing:
            raise ValueError("Missing required DATASET field(s): " + ", ".join(missing))
        base_path = Path(section["base_path"]).expanduser()
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        if not base_path.is_dir():
            raise FileNotFoundError(f"base_path does not exist: {base_path}")
        output = Path(section.get("output_dir", "output")).expanduser()
        if not output.is_absolute():
            output = base_path / output
        baseline = str(section["baseline"]).lower()
        if baseline not in list_available_baselines():
            raise ValueError(f"Unknown baseline: {baseline}")
        parameters = section.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("DATASET.parameters must be a mapping")
        evaluation = data.get("EVALUATION", {})
        if not isinstance(evaluation, dict):
            raise ValueError("EVALUATION must be a mapping")
        max_time_difference_s = float(evaluation.get("max_time_difference_s", 0.02))
        if max_time_difference_s <= 0:
            raise ValueError("EVALUATION.max_time_difference_s must be positive")
        return cls(
            base_path=base_path,
            name=str(section["name"]),
            baseline=baseline,
            dataset=str(section["dataset"]),
            output_dir=output.resolve(),
            sensor_type=str(section.get("sensor_type", "mono")),
            groundtruth_format=str(section.get("groundtruth_format", "kitti")),
            parameters=dict(parameters),
            max_time_difference_s=max_time_difference_s,
        )


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"vslamlab_custom_dataset_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import dataset module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(
    selector: str,
    config_path: str | Path | None = None,
    benchmark_path: Path | None = None,
) -> DatasetVSLAMLAB:
    """Resolve a registered dataset name or a portable Python module path."""
    if selector.lower() in list_available_datasets():
        dataset = get_dataset(selector)
        if not isinstance(dataset, DatasetVSLAMLAB):
            raise TypeError(f"Dataset registry returned an invalid object for {selector}")
        return dataset

    module_path = Path(selector).expanduser()
    candidates = [module_path]
    if not module_path.is_absolute():
        if config_path is not None:
            candidates.insert(0, Path(config_path).expanduser().resolve().parent / module_path)
        candidates.append(VSLAM_LAB_DIR / module_path)
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Dataset module does not exist: {selector}")
    module = _load_module(resolved)
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, DatasetVSLAMLAB)
        and value is not DatasetVSLAMLAB
        and value.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise ValueError(f"Expected exactly one DatasetVSLAMLAB subclass in {resolved}; found {len(classes)}")
    dataset_class = classes[0]
    parameters = inspect.signature(dataset_class.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "benchmark_path" in parameters:
        if benchmark_path is None:
            raise ValueError(f"Dataset {dataset_class.__name__} requires benchmark_path")
        kwargs["benchmark_path"] = benchmark_path
    if "dataset_name" in parameters:
        kwargs["dataset_name"] = resolved.stem.removeprefix("dataset_")
    return dataset_class(**kwargs)


@contextmanager
def headless_environment(enabled: bool) -> Iterator[None]:
    overrides = (
        {
            "DISPLAY": "",
            "QT_QPA_PLATFORM": "offscreen",
            "PANGOLIN_WINDOW_URI": "headless://",
        }
        if enabled
        else {}
    )
    original = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _SingleExperiment:
    def __init__(self, folder: Path, module: str, parameters: dict[str, Any]) -> None:
        self.folder = folder
        self.module = module
        self.parameters = parameters
        self.num_runs = 1


def _prepare(config_yaml: str | Path, headless: bool) -> tuple[SingleSequenceConfig, Any, Any, _SingleExperiment, Path]:
    config = SingleSequenceConfig.load(config_yaml)
    dataset = load_dataset(config.dataset, config_yaml, config.base_path)
    dataset.dataset_path = config.base_path
    dataset.dataset_folder = ""
    dataset.sequence_names = [config.name]
    if hasattr(dataset, "prepare_local_sequence"):
        dataset.prepare_local_sequence(config.base_path, config.name)
    sequence_path = dataset.sequence_path(config.name)
    required = [
        sequence_path / "rgb_0",
        sequence_path / "rgb.csv",
        sequence_path / "calibration.yaml",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Prepared sequence is missing: " + ", ".join(map(str, missing)))

    baseline = get_baseline(config.baseline)
    parameters = dict(baseline.get_default_parameters())
    parameters.update(config.parameters)
    parameters.update(
        {
            "mode": config.sensor_type,
            "gui": not headless,
            "headless": headless,
            "show_gui": not headless,
        }
    )
    if headless:
        parameters["verbose"] = 0
    run_root = config.output_dir / ".vslamlab_run"
    run_root.mkdir(parents=True, exist_ok=True)
    experiment = _SingleExperiment(run_root, config.baseline, parameters)
    run_folder = run_root / config.name
    return config, dataset, baseline, experiment, run_folder


def _trajectory(run_folder: Path) -> Path:
    trajectory = run_folder / f"00000_{TRAJECTORY_FILE_NAME}.csv"
    if not trajectory.exists():
        raise RuntimeError(f"Baseline did not produce a trajectory: {trajectory}")
    if len(trajectory.read_text(encoding="utf-8").splitlines()) < 2:
        raise RuntimeError(f"Baseline produced an empty trajectory: {trajectory}")
    return trajectory


def _write_report(trajectory: Path, groundtruth: Path, output: Path, title: str) -> None:
    """Write a lightweight trajectory PDF without invoking the comparison pipeline."""
    import matplotlib.pyplot as plt
    import pandas as pd

    predicted = pd.read_csv(trajectory)
    reference = pd.read_csv(groundtruth)
    if len(predicted.columns) < 4 or len(reference.columns) < 4:
        raise ValueError("Trajectory report requires timestamp and xyz columns")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, dimensions, labels in (
        (axes[0], (1, 2), ("x (m)", "y (m)")),
        (axes[1], (1, 3), ("x (m)", "z (m)")),
    ):
        axis.plot(
            reference.iloc[:, dimensions[0]],
            reference.iloc[:, dimensions[1]],
            label="ground truth",
        )
        axis.plot(
            predicted.iloc[:, dimensions[0]],
            predicted.iloc[:, dimensions[1]],
            label="estimate",
        )
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_single(config_yaml: str | Path, evaluate: bool) -> Path:
    config, dataset, baseline, experiment, run_folder = _prepare(config_yaml, headless=evaluate)
    print_msg(f"\n{SCRIPT_LABEL}", f"Running {config.baseline} on {config.name}")
    with headless_environment(evaluate):
        results = run_sequence(0, experiment, baseline, dataset, config.name)
    trajectory = _trajectory(run_folder)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trajectory, config.output_dir / trajectory.name)
    if not results.get("success", False):
        print_msg(
            f"{ws(4)}",
            "Baseline reported failure but produced a usable trajectory",
            "warning",
        )
    if not evaluate:
        return trajectory

    groundtruth = run_folder / "groundtruth.csv"
    if not groundtruth.exists():
        raise FileNotFoundError(f"Ground truth is required for evaluation: {groundtruth}")
    fast_lio_trajectory: Path | None = None
    warnings: list[str] = []
    if fast_lio_reference_enabled(config_yaml):
        print_msg(f"{ws(4)}", "Generating or reusing FAST-LIO reference")
        fast_lio_trajectory = generate_fast_lio_reference(config_yaml)
        fast_lio_manifest = fast_lio_trajectory.parent / "manifest.json"
        if fast_lio_manifest.is_file():
            shutil.copy2(fast_lio_manifest, config.output_dir / "fast_lio_manifest.json")
        warnings.append(
            "Camera/body and Ouster/body mount extrinsics are unavailable; cross-sensor "
            "comparisons use trajectory-derived alignment. Translation and especially "
            "rotation metrics are approximate until those fixed transforms are supplied."
        )
    metrics, aligned = write_pairwise_metrics(
        config.output_dir / "metrics.json",
        trajectory,
        groundtruth,
        fast_lio_trajectory,
        config.max_time_difference_s,
        config.sensor_type,
        warnings,
    )
    write_combined_report(
        aligned,
        config.output_dir / "trajectory_report.pdf",
        f"{config.baseline}: {config.name}",
    )
    print_msg(f"{ws(4)}", f"Results saved to {config.output_dir}")
    return metrics


def eval_metrics_single(config_yaml: str | Path) -> Path:
    return run_single(config_yaml, evaluate=True)


def demo_single(config_yaml: str | Path) -> Path:
    return run_single(config_yaml, evaluate=False)
