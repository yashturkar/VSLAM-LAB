import sys, os, yaml, shutil, csv, time
import pandas as pd
import importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any
from pathlib import Path
from inputimeout import inputimeout, TimeoutOccurred
 

from utilities import ws, load_yaml_file, print_msg, show_time, read_csv
from Datasets.get_dataset import list_available_datasets, get_dataset
from Baselines.get_baseline import list_available_baselines, get_baseline
from Run.run_functions import run_sequence
from Evaluate.evaluate_functions import evaluate_sequence
from Evaluate import compare_functions
from Evaluate.metrics_json import generate_metrics_json, extract_metrics_from_evaluation, compute_trajectory_metrics, compute_rotation_rmse
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION, VSLAM_LAB_DIR, CONFIG_DEFAULT, VSLAMLAB_VIDEOS, COMPARISONS_YAML_DEFAULT, TRAJECTORY_FILE_NAME, GROUNTRUTH_FILE

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

##################################################################################################################################################
# write_demo_yaml_fles
##################################################################################################################################################
##################################################################################################################################################
def write_demo_yaml_fles(baseline_name: str, dataset_name: str, sequence_name: str, mode :str = None) -> None:
    # Write experiment yaml
    exp_demo = 'demo'
    exp_yaml = VSLAM_LAB_DIR / 'configs' / 'exp_demo.yaml'
    config_yaml =  VSLAM_LAB_DIR / 'configs' / 'config_demo.yaml'

    baseline = get_baseline(baseline_name)

    exp_data = {}
    exp_data[exp_demo] = {}
    exp_data[exp_demo]['Config'] = str(config_yaml)
    exp_data[exp_demo]['NumRuns'] = 1
    exp_data[exp_demo]['Parameters'] = baseline.get_default_parameters()
    exp_data[exp_demo]['Module'] = baseline_name
    if mode:
        exp_data[exp_demo]['Parameters']['mode'] = mode
    
    # Write experiment yaml
    with open(exp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp_data, f)

    # Write config yaml
    exp_seq = { dataset_name: [sequence_name] }   
    with open(config_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp_seq, f)

##################################################################################################################################################
# Info commands
##################################################################################################################################################
##################################################################################################################################################

def baseline_info(baseline_name: str) -> None:

    baseline_list = list_available_baselines()
    if baseline_name not in baseline_list:
        print_msg(f"\n{SCRIPT_LABEL}", f"'{baseline_name}' baseline doesn't exist.", "error")
        print_baselines()
        exit(0)

    baseline = get_baseline(baseline_name)    
    baseline.info_print()

def print_baselines() -> None:
    baseline_list = list_available_baselines()
    print_msg(f"\n{SCRIPT_LABEL}", f"Accessible baselines in VSLAM-LAB:", "info")
    for baseline in baseline_list:
        print(f" - {baseline}")
    print("For detailed information about a baseline, use 'pixi run baseline-info <baseline_name>'")

def print_datasets() -> None:
    dataset_list = list_available_datasets()
    print(f"\n{SCRIPT_LABEL}Accessible datasets in VSLAM-LAB:")
    for dataset in dataset_list:
        print(f" - {dataset}")

def add_video(video_path):
    abs_path = os.path.abspath(video_path)
    video_name_ext = os.path.basename(abs_path)
    sequence_name = os.path.splitext(video_name_ext)[0]
    dataset_videos_yaml = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'dataset_videos.yaml')
    with open(dataset_videos_yaml, 'r') as f:
        data = yaml.safe_load(f)
    if 'sequence_names' not in data or data['sequence_names'] is None:
        data['sequence_names'] = []
    if sequence_name not in data['sequence_names']:
        data['sequence_names'].append(sequence_name)
    with open(dataset_videos_yaml, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
    if not os.path.exists(os.path.join(VSLAMLAB_VIDEOS, video_name_ext)):
        shutil.copy2(abs_path, os.path.join(VSLAMLAB_VIDEOS, video_name_ext))
    
    return sequence_name

##################################################################################################################################################
##################################################################################################################################################
class Experiment:
    def __init__(self, name, settings):            
        self.name = name
        self.folder = os.path.join(VSLAMLAB_EVALUATION, self.name)
        self.num_runs = settings.get('NumRuns', 1)
        self.module = settings.get('Module', "default")
        self.parameters = settings['Parameters']

        self.log_csv = os.path.join(self.folder, 'vslamlab_exp_log.csv')
        self.config_yaml = os.path.join(VSLAM_LAB_DIR, 'configs', settings.get('Config', CONFIG_DEFAULT))
        self.ablation_csv = settings.get('Ablation', None)

def load_experiments(exp_yaml: str | Path)-> list[Any]:
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    experiments = {}
    for exp_name, settings in exp_data.items():
        experiments[exp_name] = Experiment(exp_name, settings)

    return experiments

##################################################################################################################################################
# compare_exp
##################################################################################################################################################
##################################################################################################################################################
def compare_exp(exp_yaml: str | Path) -> None:

    experiments = load_experiments(exp_yaml)

    comparison_path = os.path.join(VSLAMLAB_EVALUATION, f"comp_{str(os.path.basename(exp_yaml)).replace('.yaml', '')}")
    print_msg(f"\n{SCRIPT_LABEL}", f"Comparing (in {comparison_path}) ...")
    if os.path.exists(comparison_path):
        shutil.rmtree(comparison_path)
    os.makedirs(comparison_path)
    os.makedirs(os.path.join(comparison_path, 'figures'))
    compare_functions.full_comparison(experiments, VSLAMLAB_BENCHMARK, COMPARISONS_YAML_DEFAULT, comparison_path)

##################################################################################################################################################
# eval_exp
##################################################################################################################################################
##################################################################################################################################################
def evaluate_exp(exp_yaml: str | Path, overwrite: bool = False) -> None:
    
    experiments = load_experiments(exp_yaml)
    first_evaluation_found = True
    for [_, exp] in experiments.items():
        exp_log = read_csv(exp.log_csv)
        if(not exp_log['EVALUATION'].str.contains('none').any()) and not overwrite:
            continue
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    if first_evaluation_found:
                        print_msg(f"\n{SCRIPT_LABEL}", f"Evaluating (in {VSLAMLAB_EVALUATION}) ...")
                        first_evaluation_found = False
                    evaluate_sequence(exp, dataset, sequence_name, overwrite)

##################################################################################################################################################
# eval_metrics
##################################################################################################################################################
##################################################################################################################################################
def eval_metrics(exp_yaml: str | Path) -> None:
    """
    Run SLAM, evaluate trajectories, and generate metrics.json files.
    
    This function:
    1. Runs SLAM experiments (run_exp)
    2. Evaluates trajectories (evaluate_exp)
    3. Generates metrics.json files in the evaluation folder for each sequence
    """
    # Step 1: Run SLAM
    print_msg(f"\n{SCRIPT_LABEL}", f"Running SLAM experiments: {exp_yaml}")
    run_exp(exp_yaml)
    
    # Step 2: Evaluate trajectories
    print_msg(f"\n{SCRIPT_LABEL}", f"Evaluating trajectories: {exp_yaml}")
    evaluate_exp(exp_yaml, overwrite=False)
    
    # Step 3: Generate metrics.json files
    print_msg(f"\n{SCRIPT_LABEL}", f"Generating metrics.json files: {exp_yaml}")
    experiments = load_experiments(exp_yaml)
    
    for [_, exp] in experiments.items():
        exp_log = read_csv(exp.log_csv)
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    # Find all runs for this sequence that have been evaluated
                    sequence_runs = exp_log[
                        (exp_log["sequence_name"] == sequence_name) & 
                        (exp_log["EVALUATION"] != "none") &
                        (exp_log["EVALUATION"] != "failed")
                    ]
                    
                    if sequence_runs.empty:
                        # If no evaluated runs, try to generate for successful runs anyway
                        sequence_runs = exp_log[
                            (exp_log["sequence_name"] == sequence_name) & 
                            (exp_log["SUCCESS"] == "True")
                        ]
                    
                    for _, row in sequence_runs.iterrows():
                        exp_it = str(row["exp_it"]).zfill(5)
                        # Determine status based on whether trajectory exists and can be evaluated
                        # Don't rely solely on SUCCESS field, as trajectories may exist even if SLAM marked as failed
                        trajectories_path = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
                        trajectory_csv = os.path.join(trajectories_path, f"{exp_it}_{TRAJECTORY_FILE_NAME}.csv")
                        trajectory_txt = os.path.join(trajectories_path, f"{exp_it}_{TRAJECTORY_FILE_NAME}.txt")
                        evaluation_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name, 'vslamlab_evaluation')
                        ate_csv = os.path.join(evaluation_folder, "ate.csv")
                        
                        # Check if trajectory exists and evaluation was successful
                        trajectory_exists = os.path.exists(trajectory_csv) or os.path.exists(trajectory_txt)
                        evaluation_successful = os.path.exists(ate_csv) and str(row.get("EVALUATION", "")) not in ["none", "failed"]
                        
                        # If trajectory exists and was evaluated, consider it SUCCESS
                        if trajectory_exists and evaluation_successful:
                            status = "SUCCESS"
                        elif trajectory_exists:
                            # Trajectory exists but not evaluated - still try to generate metrics
                            status = "SUCCESS"
                        else:
                            # No trajectory file - mark as FAILURE
                            status = "FAILURE"
                        
                        try:
                            metrics_json_path = generate_metrics_json(
                                exp, dataset, sequence_name, exp_it, status
                            )
                            if metrics_json_path:
                                print_msg(f"{ws(4)}", f"Generated: {metrics_json_path}", verb='LOW')
                        except Exception as e:
                            print_msg(f"{ws(4)}", f"Error generating metrics.json for {sequence_name} (exp_it={exp_it}): {e}", "error")

##################################################################################################################################################
# eval_metrics_single
##################################################################################################################################################
##################################################################################################################################################
def load_dataset_class_from_file(dataset_file_path: str | Path):
    """
    Dynamically load a dataset class from a Python file.
    
    Args:
        dataset_file_path: Path to the dataset Python file
        
    Returns:
        Dataset class
    """
    dataset_path = Path(dataset_file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("dataset_module", dataset_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {dataset_file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the dataset class (should end with _dataset)
    dataset_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            name.endswith('_dataset') and 
            hasattr(obj, '__module__') and 
            obj.__module__ == module.__name__):
            dataset_class = obj
            break
    
    if dataset_class is None:
        raise ValueError(f"Could not find dataset class in {dataset_file_path}")
    
    return dataset_class


def generate_metrics_json_single(trajectory_path, groundtruth_csv, evaluation_folder, exp_it, dataset, status="SUCCESS"):
    """
    Generate metrics.json file for single sequence evaluation.
    
    Args:
        trajectory_path: Path to trajectory CSV or TXT file
        groundtruth_csv: Path to groundtruth CSV file
        evaluation_folder: Folder where evaluation results are stored
        exp_it: Experiment iteration (e.g., "00000")
        dataset: Dataset object
        status: Status of the SLAM run ("SUCCESS" or "FAILURE")
        
    Returns:
        Path to generated metrics.json file or None if generation fails
    """
    import json
    from datetime import datetime
    
    os.makedirs(evaluation_folder, exist_ok=True)
    metrics_json_path = os.path.join(evaluation_folder, "metrics.json")
    
    trajectory_exists = os.path.exists(trajectory_path)
    
    # Initialize metrics structure
    metrics = {
        "status": status,
        "rmse": {
            "translation": None,
            "rotation": None,
            "total": None
        },
        "ate": None,
        "trajectory_length": None,
        "length_ratio": None,
        "weighted_rmse": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # If no trajectory exists, mark as FAILURE and return early
    if not trajectory_exists:
        metrics["status"] = "FAILURE"
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics_json_path
    
    # Extract ATE metrics
    ate_csv = os.path.join(evaluation_folder, "ate.csv")
    ate_metrics = extract_metrics_from_evaluation(ate_csv, exp_it)
    
    if ate_metrics:
        metrics["ate"] = ate_metrics
        metrics["rmse"]["translation"] = ate_metrics["rmse"]
        metrics["rmse"]["total"] = ate_metrics["rmse"]
        metrics["status"] = "SUCCESS"
    
    # Compute rotation RMSE
    if os.path.exists(trajectory_path) and os.path.exists(groundtruth_csv):
        rotation_metrics = compute_rotation_rmse(
            groundtruth_csv,
            trajectory_path,
            evaluation_folder,
            1.0 / dataset.rgb_hz
        )
        if rotation_metrics:
            metrics["rmse"]["rotation"] = rotation_metrics["rmse"]
    
    # Compute trajectory metrics
    trajectory_metrics = compute_trajectory_metrics(evaluation_folder, exp_it, groundtruth_csv)
    if trajectory_metrics:
        if "trajectory_length" in trajectory_metrics:
            metrics["trajectory_length"] = trajectory_metrics["trajectory_length"]
        if "length_ratio" in trajectory_metrics:
            metrics["length_ratio"] = trajectory_metrics["length_ratio"]
    
    # Calculate weighted_rmse = RMSE / C^2 where C is coverage (length_ratio)
    if metrics["rmse"]["translation"] is not None and metrics["length_ratio"] is not None:
        if metrics["length_ratio"] > 0:
            metrics["weighted_rmse"] = float(metrics["rmse"]["translation"] / (metrics["length_ratio"] ** 2))
    
    # Final check: if we have any meaningful metrics, mark as SUCCESS
    if metrics["ate"] is not None or metrics["rmse"]["translation"] is not None:
        metrics["status"] = "SUCCESS"
    elif metrics["trajectory_length"] is not None:
        metrics["status"] = "SUCCESS"
    
    # Write metrics.json
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_json_path


def eval_metrics_single(config_yaml: str | Path) -> None:
    """
    Run SLAM, evaluate trajectory, and generate metrics.json for a single sequence.
    
    This function:
    1. Loads a single sequence config
    2. Dynamically loads the dataset class
    3. Runs SLAM
    4. Evaluates the trajectory
    5. Generates metrics.json and copies trajectory files to output_dir
    """
    print_msg(f"\n{SCRIPT_LABEL}", f"Processing single sequence config: {config_yaml}")
    
    # Load config
    config_data = load_yaml_file(config_yaml)
    if "DATASET" not in config_data:
        print_msg(SCRIPT_LABEL, "Error: Config must contain DATASET section", "error")
        sys.exit(1)
    
    dataset_config = config_data["DATASET"]
    base_path = Path(dataset_config["base_path"])
    sequence_name = dataset_config["name"]
    baseline_name = dataset_config["baseline"]
    output_dir = dataset_config.get("output_dir", "output")
    sensor_type = dataset_config.get("sensor_type", "mono")
    dataset_file = Path(dataset_config["dataset"])
    
    # Validate paths
    if not base_path.exists():
        print_msg(SCRIPT_LABEL, f"Error: base_path does not exist: {base_path}", "error")
        sys.exit(1)
    
    if not dataset_file.exists():
        print_msg(SCRIPT_LABEL, f"Error: dataset file does not exist: {dataset_file}", "error")
        sys.exit(1)
    
    # Dynamically load dataset class
    print_msg(f"{SCRIPT_LABEL}", f"Loading dataset class from: {dataset_file}")
    try:
        DatasetClass = load_dataset_class_from_file(dataset_file)
    except Exception as e:
        print_msg(SCRIPT_LABEL, f"Error loading dataset class: {e}", "error")
        sys.exit(1)
    
    # Instantiate dataset with base_path as benchmark_path
    # Check if dataset class needs dataset_name parameter
    import inspect
    sig = inspect.signature(DatasetClass.__init__)
    params = list(sig.parameters.keys())
    
    # Extract dataset name from class name (e.g., LIGHTNING_dataset -> "lightning")
    dataset_name_from_class = DatasetClass.__name__.lower().replace('_dataset', '')
    
    if 'dataset_name' in params:
        # LIGHTNING_dataset and similar classes
        dataset = DatasetClass(benchmark_path=base_path, dataset_name=dataset_name_from_class)
    else:
        # Standard dataset classes - need to pass dataset_name as first arg
        dataset = DatasetClass(dataset_name_from_class, base_path)
    
    # Override dataset_path to point to base_path directly for single sequence
    # The sequence data is in base_path, not in base_path/DATASET_FOLDER
    dataset.dataset_path = base_path
    
    # Get baseline
    baseline = get_baseline(baseline_name)
    baseline.check_installation()
    
    # Create a minimal experiment-like object
    class SingleExperiment:
        def __init__(self, folder, parameters=None):
            self.folder = folder
            self.parameters = parameters or {}
            self.num_runs = 1
            self.module = baseline_name
    
    # Detect streamlit structure: image_0/, sequences/, poses/, config.yaml
    image_0_path = base_path / "image_0"
    sequences_path = base_path / "sequences"
    poses_path = base_path / "poses"
    config_yaml_path = base_path / "config.yaml"
    
    is_streamlit_structure = (image_0_path.exists() and sequences_path.exists() and 
                              poses_path.exists() and config_yaml_path.exists())
    
    # Create sequence folder structure that baseline expects
    # Baseline constructs: exp.folder / dataset.dataset_folder / sequence_name
    # We want: base_path / sequence_name
    # So set exp.folder = base_path, and baseline will create base_path / LIGHTNING / sequence_name
    # But we actually want base_path / sequence_name, so we need to handle this differently
    # For single sequence, we'll set exp.folder to base_path and override dataset.dataset_folder
    sequence_path = base_path / sequence_name
    sequence_path.mkdir(parents=True, exist_ok=True)
    
    # Set exp.folder to base_path, and the baseline will construct the path
    # But we need to ensure files are in the right place
    # Actually, let's set exp.folder to base_path and override dataset_folder to empty
    exp_folder_base = base_path
    
    # Disable GUI for eval-metrics-single mode
    exp_parameters = {"mode": sensor_type, "gui": False, "headless": True, "show_gui": False}
    exp = SingleExperiment(folder=str(exp_folder_base), parameters=exp_parameters)
    
    # The baseline constructs exp_folder as: exp.folder / dataset.dataset_folder / sequence_name
    # We want: base_path / sequence_name
    # So we need to set exp.folder = base_path and dataset.dataset_folder = "" 
    # But os.path.join("base_path", "", "sequence_name") = "base_path/sequence_name" ✓
    original_dataset_folder = dataset.dataset_folder
    dataset.dataset_folder = ""  # Empty so path becomes base_path/sequence_name instead of base_path/LIGHTNING/sequence_name
    
    # The actual exp_folder where baseline will save files
    # Baseline will construct: os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    # = os.path.join(base_path, "", sequence_name) = base_path/sequence_name
    actual_exp_folder = base_path / sequence_name
    
    # Run SLAM
    print_msg(f"\n{SCRIPT_LABEL}", f"Running SLAM: {baseline.label} on {sequence_name}")
    exp_it = 0
    
    # Handle streamlit structure
    if is_streamlit_structure:
        print_msg(f"{SCRIPT_LABEL}", f"Detected streamlit structure, converting files...")
        
        # Find sequence folder in sequences/ (might be different from config name)
        sequence_folders = [f.name for f in sequences_path.iterdir() if f.is_dir()]
        if sequence_folders:
            actual_sequence_name = sequence_folders[0]  # Use first found sequence folder
            times_txt = sequences_path / actual_sequence_name / "times.txt"
            poses_txt = poses_path / f"{actual_sequence_name}.txt"
        else:
            # Fallback: try using sequence_name from config
            times_txt = sequences_path / sequence_name / "times.txt"
            poses_txt = poses_path / f"{sequence_name}.txt"
        
        # Create rgb_0 folder and copy/link images
        rgb_0_path = sequence_path / "rgb_0"
        if not rgb_0_path.exists():
            rgb_0_path.mkdir(parents=True, exist_ok=True)
            # Copy images from image_0 to rgb_0
            if image_0_path.exists():
                for img_file in image_0_path.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        shutil.copy2(img_file, rgb_0_path / img_file.name)
        
        # Create rgb.csv from image_0 and times.txt
        if times_txt.exists() and rgb_0_path.exists():
            rgb_csv = sequence_path / "rgb.csv"
            times = []
            with open(times_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        times.append(float(line))
            
            # Collect and sort image filenames
            rgb_files = sorted([f.name for f in rgb_0_path.iterdir() 
                              if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            
            # Write CSV
            with open(rgb_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ts_rgb0 (s)', 'path_rgb0'])
                for t, fname in zip(times, rgb_files):
                    writer.writerow([f"{t:.6f}", f"rgb_0/{fname}"])
            print_msg(f"{ws(4)}", f"Created rgb.csv from {len(rgb_files)} images")
        
        # Create calibration.yaml from config.yaml using proper OpenCV FileStorage format
        if config_yaml_path.exists():
            calibration_yaml = sequence_path / "calibration.yaml"
            # Try to load and convert config.yaml
            try:
                with open(config_yaml_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract camera parameters from config.yaml
                # Handle both nested Camera dict and flat Camera.fx format
                if 'Camera' in config_data:
                    cam_data = config_data['Camera']
                else:
                    # Try flat format like Camera.fx
                    cam_data = {}
                    for key, value in config_data.items():
                        if key.startswith('Camera.'):
                            cam_data[key.replace('Camera.', '')] = value
                
                # Convert to VSLAM-LAB calibration format
                camera0 = {
                    "model": "OPENCV",
                    "fx": float(cam_data.get('fx', config_data.get('Camera.fx', 1446.91))),
                    "fy": float(cam_data.get('fy', config_data.get('Camera.fy', 1451.58))),
                    "cx": float(cam_data.get('cx', config_data.get('Camera.cx', 964.94))),
                    "cy": float(cam_data.get('cy', config_data.get('Camera.cy', 607.07))),
                    "k1": float(cam_data.get('k1', config_data.get('Camera.k1', -0.139))),
                    "k2": float(cam_data.get('k2', config_data.get('Camera.k2', 0.237))),
                    "p1": float(cam_data.get('p1', config_data.get('Camera.p1', -0.00064))),
                    "p2": float(cam_data.get('p2', config_data.get('Camera.p2', 0.00071))),
                    "k3": float(cam_data.get('k3', config_data.get('Camera.k3', -0.273)))
                }
                
                # Use dataset's write_calibration_yaml method to create proper OpenCV FileStorage format
                dataset.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0)
                print_msg(f"{ws(4)}", f"Created calibration.yaml from config.yaml using OpenCV FileStorage format")
            except Exception as e:
                print_msg(f"{ws(4)}", f"Error creating calibration.yaml: {e}", "error")
                # Try to use dataset's method with defaults
                try:
                    camera0 = {
                        "model": "OPENCV",
                        "fx": 1446.91, "fy": 1451.58, "cx": 964.94, "cy": 607.07,
                        "k1": -0.139, "k2": 0.237, "p1": -0.00064, "p2": 0.00071, "k3": -0.273
                    }
                    dataset.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0)
                    print_msg(f"{ws(4)}", f"Created calibration.yaml with default parameters")
                except Exception as e2:
                    print_msg(f"{ws(4)}", f"Failed to create calibration.yaml: {e2}", "error")
                    sys.exit(1)
        
        # Create groundtruth.csv from poses/{sequence_name}.txt and times.txt
        if poses_txt.exists() and times_txt.exists():
            groundtruth_csv = sequence_path / GROUNTRUTH_FILE
            times = []
            with open(times_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        times.append(float(line))
            
            with open(poses_txt, 'r') as src, open(groundtruth_csv, 'w', newline='') as dst:
                writer = csv.writer(dst)
                writer.writerow(['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
                
                for idx, line in enumerate(src):
                    if idx >= len(times):
                        break
                    vals = list(map(float, line.strip().split()))
                    # row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                    Rm = np.array([[vals[0], vals[1], vals[2]],
                                [vals[4], vals[5], vals[6]],
                                [vals[8], vals[9], vals[10]]], dtype=float)
                    tx, ty, tz = vals[3], vals[7], vals[11]
                    qx, qy, qz, qw = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
                    ts = times[idx]
                    writer.writerow([f"{ts:.6f}", tx, ty, tz, qx, qy, qz, qw])
            print_msg(f"{ws(4)}", f"Created groundtruth.csv from poses and times")
    
    # Standard structure: look for existing files
    rgb_csv = sequence_path / "rgb.csv"
    if not rgb_csv.exists():
        rgb_csv = sequence_path / "rgb_0.csv"
    
    if not rgb_csv.exists():
        print_msg(SCRIPT_LABEL, f"Error: rgb.csv not found in {sequence_path}", "error")
        sys.exit(1)
    
    # Baseline will construct exp_folder as base_path / sequence_name (since dataset_folder is empty)
    actual_exp_folder = base_path / sequence_name
    rgb_exp_csv = actual_exp_folder / "rgb_exp.csv"
    if rgb_csv.exists():
        shutil.copy(rgb_csv, rgb_exp_csv)
    
    # Copy groundtruth if needed
    groundtruth_src = sequence_path / GROUNTRUTH_FILE
    groundtruth_dst = actual_exp_folder / GROUNTRUTH_FILE
    if groundtruth_src.exists() and not groundtruth_dst.exists():
        shutil.copy(groundtruth_src, groundtruth_dst)
    
    # Build and execute command
    # The baseline will construct sequence_path as dataset.dataset_path / sequence_name
    # which will be base_path / sequence_name
    # But we need to make sure the actual data files are accessible
    # If data is in base_path directly, we need to handle this differently
    # For now, we'll assume data is in base_path/sequence_name or we create symlinks/copies
    
    # Disable GUI by setting DISPLAY environment variable to empty
    # This prevents GUI windows from opening in eval-metrics-single mode
    original_display = os.environ.get('DISPLAY', None)
    os.environ['DISPLAY'] = ''
    
    try:
        exec_command = baseline.build_execute_command(exp_it, exp, dataset, sequence_name)
        
        # Debug: Print command and paths
        # Note: baseline will construct exp_folder as exp.folder / dataset.dataset_folder / sequence_name
        # Since we set dataset.dataset_folder = "", it becomes base_path / sequence_name
        actual_exp_folder = base_path / sequence_name
        print_msg(f"{ws(4)}", f"Command: {exec_command}", verb='LOW')
        print_msg(f"{ws(4)}", f"Exp folder base: {exp_folder_base}", verb='LOW')
        print_msg(f"{ws(4)}", f"Actual exp folder (where trajectory will be): {actual_exp_folder}", verb='LOW')
        print_msg(f"{ws(4)}", f"Sequence path: {sequence_path}", verb='LOW')
        
        # Debug: Check required files exist
        calibration_yaml = sequence_path / "calibration.yaml"
        rgb_exp_csv = actual_exp_folder / "rgb_exp.csv"
        print_msg(f"{ws(4)}", f"Calibration YAML exists: {calibration_yaml.exists()} ({calibration_yaml})", verb='LOW')
        print_msg(f"{ws(4)}", f"RGB CSV exists: {rgb_exp_csv.exists()} ({rgb_exp_csv})", verb='LOW')
        
        if calibration_yaml.exists():
            with open(calibration_yaml, 'r') as f:
                calib_content = f.read()[:200]  # First 200 chars
                print_msg(f"{ws(4)}", f"Calibration YAML preview: {calib_content}...", verb='LOW')
        
        print(f"\n{SCRIPT_LABEL}Running {baseline.label} on {sequence_name}...")
        # The baseline.build_execute_command() constructs exp_folder internally
        # We need to extract it from the command or construct it the same way
        # exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        # = os.path.join(base_path, "", sequence_name) = base_path/sequence_name
        baseline_exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        results = baseline.execute(exec_command, exp_it, baseline_exp_folder)
        
        # Debug: Print execution results
        print_msg(f"{ws(4)}", f"Execution success: {results.get('success', 'N/A')}", verb='LOW')
        print_msg(f"{ws(4)}", f"Execution comments: {results.get('comments', 'N/A')}", verb='LOW')
        
    finally:
        # Restore original DISPLAY value if it existed
        if original_display is not None:
            os.environ['DISPLAY'] = original_display
        elif 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
    
    # Check for trajectory files - baseline may save TXT instead of CSV
    # Convert TXT to CSV if needed before checking success
    baseline_exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    trajectory_csv_path = os.path.join(baseline_exp_folder, f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.csv")
    trajectory_txt_path = os.path.join(baseline_exp_folder, f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.txt")
    
    # If CSV doesn't exist but TXT does, convert it
    if not os.path.exists(trajectory_csv_path) and os.path.exists(trajectory_txt_path):
        print_msg(f"{ws(4)}", f"Converting trajectory TXT to CSV: {trajectory_txt_path}")
        from utilities import read_trajectory_txt, save_trajectory_csv
        try:
            traj_df = read_trajectory_txt(trajectory_txt_path)
            if traj_df is not None and not traj_df.empty:
                if len(traj_df.columns) >= 8:
                    traj_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
                save_trajectory_csv(trajectory_csv_path, traj_df, header=True)
                print_msg(f"{ws(4)}", f"Successfully converted TXT to CSV")
                # If conversion succeeded, mark as success even if baseline marked it as failed
                if not results['success']:
                    print_msg(f"{ws(4)}", f"Note: Baseline marked as failed (no CSV), but TXT was successfully converted", "warning")
        except Exception as e:
            print_msg(f"{ws(4)}", f"Failed to convert TXT to CSV: {e}", "error")
    
    if not results['success'] and not os.path.exists(trajectory_csv_path):
        error_msg = results.get('comments', 'Unknown error')
        print_msg(SCRIPT_LABEL, f"SLAM execution failed: {error_msg}", "error")
        
        # Debug: Check log file for errors
        actual_exp_folder = base_path / sequence_name
        log_file = actual_exp_folder / f"system_output_{str(exp_it).zfill(5)}.txt"
        if log_file.exists():
            print_msg(f"{ws(4)}", f"Checking log file: {log_file}", "error")
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    # Print last 500 characters (likely contains error)
                    if len(log_content) > 500:
                        print_msg(f"{ws(4)}", f"Last 500 chars of log:\n{log_content[-500:]}", "error")
                    else:
                        print_msg(f"{ws(4)}", f"Full log content:\n{log_content}", "error")
            except Exception as e:
                print_msg(f"{ws(4)}", f"Could not read log file: {e}", "error")
        
        # Debug: Check if trajectory file was created despite failure
        actual_exp_folder = base_path / sequence_name
        trajectory_csv = actual_exp_folder / f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.csv"
        trajectory_txt = actual_exp_folder / f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.txt"
        print_msg(f"{ws(4)}", f"Trajectory CSV exists: {trajectory_csv.exists()}", verb='LOW')
        print_msg(f"{ws(4)}", f"Trajectory TXT exists: {trajectory_txt.exists()}", verb='LOW')
        
        sys.exit(1)
    
    # Find trajectory file - baseline saves to base_path / sequence_name
    actual_exp_folder = base_path / sequence_name
    trajectory_csv = actual_exp_folder / f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.csv"
    trajectory_txt = actual_exp_folder / f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.txt"
    
    trajectory_file = None
    if trajectory_csv.exists():
        trajectory_file = str(trajectory_csv)
    elif trajectory_txt.exists():
        # Convert TXT to CSV if needed
        from utilities import read_trajectory_txt, save_trajectory_csv
        traj_df = read_trajectory_txt(str(trajectory_txt))
        if traj_df is not None and not traj_df.empty:
            if len(traj_df.columns) >= 8:
                traj_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
            save_trajectory_csv(str(trajectory_csv), traj_df, header=True)
            trajectory_file = str(trajectory_csv)
    
    if trajectory_file is None:
        print_msg(SCRIPT_LABEL, "Error: Trajectory file not found after SLAM execution", "error")
        sys.exit(1)
    
    # Evaluate trajectory
    print_msg(f"\n{SCRIPT_LABEL}", f"Evaluating trajectory: {trajectory_file}")
    
    # Find groundtruth - use sequence_path where we created it
    groundtruth_csv = sequence_path / GROUNTRUTH_FILE
    
    if not groundtruth_csv.exists():
        print_msg(SCRIPT_LABEL, f"Error: Groundtruth file not found: {groundtruth_csv}", "error")
        sys.exit(1)
    
    # Create evaluation folder - use sequence_path
    evaluation_folder = sequence_path / "vslamlab_evaluation"
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    
    # Set evo config (required for evo_ape to work correctly)
    # This matches what evaluate_sequence does
    import subprocess
    command = "pixi run -e vslamlab evo_config set save_traj_in_zip true"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if evaluation already exists (from previous run)
    traj_zip = evaluation_folder / f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.zip"
    accuracy_csv = evaluation_folder / "ate.csv"
    
    # Run evo_metric (skip if already evaluated)
    from Evaluate.evo_functions import evo_metric, evo_get_accuracy
    
    if traj_zip.exists():
        print_msg(f"{ws(4)}", f"Evaluation zip file already exists, checking for accuracy CSV...", verb='LOW')
        if accuracy_csv.exists():
            print_msg(f"{ws(4)}", f"Evaluation already complete, skipping evo_ape...", verb='LOW')
            success = [True, "Evaluation already completed"]
        else:
            # Zip exists but ate.csv doesn't - need to call evo_get_accuracy to create it
            print_msg(f"{ws(4)}", f"Zip file exists but ate.csv missing, processing zip file...", verb='LOW')
            zip_file = str(traj_zip)
            evo_get_accuracy([zip_file], str(accuracy_csv))
            success = [True, "Evaluation completed from existing zip"]
    else:
        print_msg(f"{ws(4)}", f"Running evo_ape evaluation (this may take a few minutes)...", verb='LOW')
        print_msg(f"{ws(4)}", f"Groundtruth: {groundtruth_csv}", verb='LOW')
        print_msg(f"{ws(4)}", f"Trajectory: {trajectory_file}", verb='LOW')
        print_msg(f"{ws(4)}", f"Max time diff: {1.0 / dataset.rgb_hz:.6f}s", verb='LOW')
        
        success = evo_metric('ate', str(groundtruth_csv), trajectory_file, str(evaluation_folder), 1.0 / dataset.rgb_hz)
        
        # After evo_metric succeeds, collect zip file and call evo_get_accuracy (matching evaluate-exp)
        if success[0]:
            zip_file = os.path.join(str(evaluation_folder), f"{str(exp_it).zfill(5)}_{TRAJECTORY_FILE_NAME}.zip")
            if os.path.exists(zip_file):
                print_msg(f"{ws(4)}", f"Processing zip file to create accuracy CSV...", verb='LOW')
                evo_get_accuracy([zip_file], str(accuracy_csv))
            else:
                print_msg(f"{ws(4)}", f"Warning: Zip file not found after evo_metric success: {zip_file}", "warning")
    
    if not success[0]:
        error_msg = success[1] if len(success) > 1 else 'Unknown error'
        print_msg(SCRIPT_LABEL, f"Evaluation failed: {error_msg}", "error")
        # Continue anyway to try generating metrics from what we have
        print_msg(f"{ws(4)}", f"Continuing with metrics generation despite evaluation failure...", "warning")
    else:
        print_msg(f"{ws(4)}", f"Evaluation completed successfully", verb='LOW')
    
    # Generate metrics.json
    print_msg(f"\n{SCRIPT_LABEL}", f"Generating metrics.json")
    metrics_json_path = generate_metrics_json_single(
        trajectory_file,
        str(groundtruth_csv),
        str(evaluation_folder),
        str(exp_it).zfill(5),
        dataset,
        "SUCCESS" if success[0] else "FAILURE"
    )
    
    # Output results to base_path + output_dir
    output_path = base_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print_msg(f"\n{SCRIPT_LABEL}", f"Copying results to: {output_path}")
    
    # Copy metrics.json
    output_metrics_json = output_path / "metrics.json"
    shutil.copy(metrics_json_path, output_metrics_json)
    print_msg(f"{ws(4)}", f"Copied metrics.json to {output_metrics_json}")
    
    # Copy trajectory CSV files
    if trajectory_csv.exists():
        output_trajectory = output_path / trajectory_csv.name
        shutil.copy(trajectory_csv, output_trajectory)
        print_msg(f"{ws(4)}", f"Copied trajectory to {output_trajectory}")
    
    # Generate PDF report similar to compare-exp
    print_msg(f"\n{SCRIPT_LABEL}", f"Generating PDF report...")
    try:
        # Use aligned trajectories from evaluation folder if available
        aligned_traj_file = Path(evaluation_folder) / f"{str(exp_it).zfill(5)}_KeyFrameTrajectory.csv"
        aligned_gt_file = Path(evaluation_folder) / f"{str(exp_it).zfill(5)}_gt.csv"
        
        # Use aligned files if they exist, otherwise use original files
        traj_for_plot = str(aligned_traj_file) if aligned_traj_file.exists() else trajectory_file
        gt_for_plot = str(aligned_gt_file) if aligned_gt_file.exists() else str(groundtruth_csv)
        
        report_path = output_path / "trajectory_report.pdf"
        _generate_simple_pdf_report(
            traj_for_plot,
            gt_for_plot,
            str(evaluation_folder),
            sequence_name,
            baseline_name,
            str(report_path)
        )
        print_msg(f"{ws(4)}", f"PDF report saved to: {report_path}")
    except Exception as e:
        print_msg(f"{ws(4)}", f"Could not generate PDF report: {e}", "warning")
        import traceback
        traceback.print_exc()
    
    print_msg(f"\n{SCRIPT_LABEL}", f"Single sequence evaluation complete!")
    print_msg(f"{ws(4)}", f"Results saved to: {output_path}")

def _generate_simple_pdf_report(trajectory_file, groundtruth_file, evaluation_folder, 
                                sequence_name, baseline_name, output_pdf):
    """Generate a simple PDF report with trajectory plots for a single sequence."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from utilities import read_trajectory_csv
    
    # Read trajectories - handle both comma and space-separated formats
    # Try space-separated first (TUM format), then comma-separated
    traj_df = None
    try:
        # Try space-separated first (common for TUM format files with .csv extension)
        traj_df = pd.read_csv(trajectory_file, sep=' ', header=None)
        if len(traj_df.columns) >= 8:
            traj_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'] + list(traj_df.columns[8:])
        elif len(traj_df.columns) >= 3:
            # At least assign ts, tx, ty
            new_cols = ['ts', 'tx', 'ty'] + [f'col_{i}' for i in range(3, len(traj_df.columns))]
            traj_df.columns = new_cols[:len(traj_df.columns)]
    except Exception:
        # If space-separated fails, try comma-separated
        try:
            traj_df = pd.read_csv(trajectory_file, sep=',', header=None)
            if len(traj_df.columns) >= 8:
                traj_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'] + list(traj_df.columns[8:])
            elif len(traj_df.columns) >= 3:
                new_cols = ['ts', 'tx', 'ty'] + [f'col_{i}' for i in range(3, len(traj_df.columns))]
                traj_df.columns = new_cols[:len(traj_df.columns)]
        except Exception:
            # Last resort: try with header
            try:
                traj_df = pd.read_csv(trajectory_file)
                # If we still have issues, check if columns need renaming
                if not any(col in traj_df.columns for col in ['ts', 'tx', 'ty']):
                    if len(traj_df.columns) >= 8:
                        traj_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'] + list(traj_df.columns[8:])
                    elif len(traj_df.columns) >= 3:
                        new_cols = ['ts', 'tx', 'ty'] + [f'col_{i}' for i in range(3, len(traj_df.columns))]
                        traj_df.columns = new_cols[:len(traj_df.columns)]
            except Exception as e:
                raise ValueError(f"Failed to read trajectory file: {trajectory_file}. Error: {e}")
    
    if traj_df is None or traj_df.empty:
        raise ValueError(f"Failed to read trajectory file or file is empty: {trajectory_file}")
    
    # Read groundtruth - handle both comma and space-separated formats
    gt_df = None
    try:
        # Try space-separated first (TUM format)
        gt_df = pd.read_csv(groundtruth_file, sep=' ', header=None)
        if len(gt_df.columns) >= 8:
            gt_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'] + list(gt_df.columns[8:])
        elif len(gt_df.columns) >= 3:
            new_cols = ['ts', 'tx', 'ty'] + [f'col_{i}' for i in range(3, len(gt_df.columns))]
            gt_df.columns = new_cols[:len(gt_df.columns)]
    except Exception:
        # If space-separated fails, try comma-separated
        try:
            gt_df = pd.read_csv(groundtruth_file, sep=',', header=None)
            if len(gt_df.columns) >= 8:
                gt_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'] + list(gt_df.columns[8:])
            elif len(gt_df.columns) >= 3:
                new_cols = ['ts', 'tx', 'ty'] + [f'col_{i}' for i in range(3, len(gt_df.columns))]
                gt_df.columns = new_cols[:len(gt_df.columns)]
        except Exception:
            # Last resort: try with header
            try:
                gt_df = pd.read_csv(groundtruth_file)
                # If we still have issues, check if columns need renaming
                if not any(col in gt_df.columns for col in ['ts', 'tx', 'ty']):
                    if len(gt_df.columns) >= 8:
                        gt_df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'] + list(gt_df.columns[8:])
                    elif len(gt_df.columns) >= 3:
                        new_cols = ['ts', 'tx', 'ty'] + [f'col_{i}' for i in range(3, len(gt_df.columns))]
                        gt_df.columns = new_cols[:len(gt_df.columns)]
            except Exception as e:
                raise ValueError(f"Failed to read groundtruth file: {groundtruth_file}. Error: {e}")
    
    if gt_df is None or gt_df.empty:
        raise ValueError(f"Failed to read groundtruth file or file is empty: {groundtruth_file}")
    
    # Ensure we have tx and ty columns
    # Check and normalize column names - standard format: ts, tx, ty, tz, qx, qy, qz, qw
    if 'tx' not in traj_df.columns:
        # Try to find position columns by index (standard format)
        if len(traj_df.columns) >= 3:
            traj_df['tx'] = traj_df.iloc[:, 1]  # Second column should be tx
            traj_df['ty'] = traj_df.iloc[:, 2]  # Third column should be ty
        else:
            raise ValueError(f"Could not determine position columns in trajectory file. Columns: {traj_df.columns.tolist()}, Shape: {traj_df.shape}")
    
    if 'tx' not in gt_df.columns:
        # Try to find position columns by index (standard format)
        if len(gt_df.columns) >= 3:
            gt_df['tx'] = gt_df.iloc[:, 1]  # Second column should be tx
            gt_df['ty'] = gt_df.iloc[:, 2]  # Third column should be ty
        else:
            raise ValueError(f"Could not determine position columns in groundtruth file. Columns: {gt_df.columns.tolist()}, Shape: {gt_df.shape}")
    
    # Convert to numeric, handling any string values
    for col in ['tx', 'ty']:
        if col in traj_df.columns:
            traj_df[col] = pd.to_numeric(traj_df[col], errors='coerce')
        if col in gt_df.columns:
            gt_df[col] = pd.to_numeric(gt_df[col], errors='coerce')
    
    # Drop any rows with NaN in tx or ty
    traj_df = traj_df.dropna(subset=['tx', 'ty'])
    gt_df = gt_df.dropna(subset=['tx', 'ty'])
    
    # Read metrics if available
    metrics_file = Path(evaluation_folder) / "stats_final.json"
    metrics = {}
    if metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    with PdfPages(output_pdf) as pdf:
        # Page 1: 2D Trajectory Plot
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        # Plot ground truth
        ax.plot(gt_df['tx'], gt_df['ty'], 'k-', label='Ground Truth', linewidth=2)
        
        # Plot estimated trajectory
        ax.plot(traj_df['tx'], traj_df['ty'], 'r-', label=f'{baseline_name}', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Trajectory Comparison: {sequence_name}\n{baseline_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Metrics Summary
        if metrics:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Create metrics table
            metrics_text = f"""
            Sequence: {sequence_name}
            Baseline: {baseline_name}
            
            Metrics:
            - RMSE (ATE): {metrics.get('rmse', 'N/A'):.4f} m
            - Mean Error: {metrics.get('mean', 'N/A'):.4f} m
            - Median Error: {metrics.get('median', 'N/A'):.4f} m
            - Std Dev: {metrics.get('std', 'N/A'):.4f} m
            - Min Error: {metrics.get('min', 'N/A'):.4f} m
            - Max Error: {metrics.get('max', 'N/A'):.4f} m
            - Tracked Frames: {metrics.get('num_tracked_frames', 'N/A')}
            - Evaluated Frames: {metrics.get('num_evaluated_frames', 'N/A')}
            """
            
            ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    return output_pdf

##################################################################################################################################################
# run_exp
##################################################################################################################################################
##################################################################################################################################################
def run_exp(exp_yaml: str | Path) -> None:
    start_time = time.time()

    experiments = load_experiments(exp_yaml)

    all_experiments_completed = {exp_name: False for exp_name in experiments}
    completed_runs = {}
    not_completed_runs = {}
    num_executed_runs = 0
    duration_time_total = 0
    while not all(all_experiments_completed.values()):

        remaining_iterations = 0
        for [exp_name, exp] in experiments.items():
            exp_log = read_csv(exp.log_csv)
            completed_runs[exp_name] = (exp_log["STATUS"] == "completed").sum()  
            not_completed_runs[exp_name] = (exp_log["STATUS"] != "completed").sum() 
            remaining_iterations += not_completed_runs[exp_name]
            
            if not_completed_runs[exp_name] == 0:
                all_experiments_completed[exp_name] = True
                continue
            
            first_not_finished_experiment = exp_log[exp_log["STATUS"] != "completed"].index.min()
            row = exp_log.loc[first_not_finished_experiment]
            baseline = get_baseline(row['method_name'])
            dataset = get_dataset(row['dataset_name'], VSLAMLAB_BENCHMARK)    

            if num_executed_runs == 0:
                print(f"\n{SCRIPT_LABEL}Running experiments (in {exp_yaml}) ...")
            results = run_sequence(row['exp_it'], exp, baseline, dataset, row['sequence_name'])

            duration_time = results['duration_time']
            duration_time_total += duration_time
            num_executed_runs += 1
            remaining_iterations -= 1

            exp_log["STATUS"] = exp_log["STATUS"].astype(str)
            exp_log["SUCCESS"] = exp_log["SUCCESS"].astype(str)
            exp_log["COMMENTS"] = exp_log["COMMENTS"].astype(str)
            exp_log.loc[first_not_finished_experiment, "STATUS"] = "completed"
            exp_log.loc[first_not_finished_experiment, "SUCCESS"] = results['success']
            exp_log.loc[first_not_finished_experiment, "COMMENTS"] = results['comments']
            exp_log.loc[first_not_finished_experiment, "TIME"] = duration_time
            exp_log.loc[first_not_finished_experiment, "RAM"] = results['ram']
            exp_log.loc[first_not_finished_experiment, "SWAP"] = results['swap']
            exp_log.loc[first_not_finished_experiment, "GPU"] = results['gpu']
            exp_log.to_csv(exp.log_csv, index=False)
                
            all_experiments_completed[exp_name] = exp_log['STATUS'].eq("completed").all()
        
        if(duration_time_total > 1):
            print(f"\n{SCRIPT_LABEL}: Experiment report: {exp_yaml}")
            print(f"{ws(4)}\033[93mNumber of executed iterations: {num_executed_runs} / {num_executed_runs + remaining_iterations} \033[0m")
            print(f"{ws(4)}\033[93mAverage time per iteration: {show_time(duration_time_total / num_executed_runs)}\033[0m")
            print(f"{ws(4)}\033[93mTotal time consumed: {show_time(duration_time_total)}\033[0m")
            print(f"{ws(4)}\033[93mRemaining time until completion: {show_time(remaining_iterations * duration_time_total / num_executed_runs)}\033[0m")

    if num_executed_runs > 0:
        run_time = (time.time() - start_time)
        print(f"\033[93m[Experiment runtime: {show_time(run_time)}]\033[0m")

##################################################################################################################################################
# download_sequence
##################################################################################################################################################
##################################################################################################################################################

def download_sequence(dataset_name: str, sequence_name: str) -> None:
    dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
    dataset.download_sequence(sequence_name)

def download_sequences(dataset_sequence_name: list[str]) -> None:
    for i in range(0, len(dataset_sequence_name), 2):
        dataset_name = dataset_sequence_name[i]
        sequence_name = dataset_sequence_name[i + 1]
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        dataset.download_sequence(sequence_name)

def download_dataset(dataset_name: str) -> None:
    dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
    for sequence_name in dataset.get_sequence_names():
        dataset.download_sequence(sequence_name)

def download_datasets(dataset_names: list[str]) -> None:
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for sequence_name in dataset.get_sequence_names():
            dataset.download_sequence(sequence_name)

##################################################################################################################################################
# install_baseline
##################################################################################################################################################
##################################################################################################################################################

def install_baseline(baseline_name: list[str]) -> None:
    baseline = get_baseline(baseline_name)
    is_baseline_installed, _ = baseline.is_installed()
    if not is_baseline_installed:
        baseline.git_clone()
        baseline.install()

def install_baselines(baselines_to_install: str) -> None:
    for baseline_name in baselines_to_install:
        install_baseline(baseline_name)

##################################################################################################################################################
# check_experiment_state
##################################################################################################################################################
##################################################################################################################################################


def check_experiment_state(exp_yaml: str | Path) -> None:
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment running state: {exp_yaml}", verb='LOW')

    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    total_num_runs = 0
    executed_num_runs = 0
    
    for exp_name, settings in exp_data.items():
        exp_folder = VSLAMLAB_EVALUATION / exp_name
        exp_log_csv = exp_folder / "vslamlab_exp_log.csv"

        total_num_runs_exp = settings.get('NumRuns')
        total_num_runs += total_num_runs_exp

        executed_num_runs_exp = 0
        if exp_folder.exists() & exp_log_csv.exists():
            exp_log = pd.read_csv(exp_log_csv)
            executed_num_runs_exp += (exp_log["STATUS"] == "completed").sum()  
            executed_num_runs += executed_num_runs_exp
        
        if executed_num_runs_exp == total_num_runs_exp:
            print(f"{ws(4)}- {exp_name}: \033[92m{executed_num_runs_exp} / {total_num_runs_exp} ({100 * executed_num_runs_exp/total_num_runs_exp} %)\033[0m")
        else:
            print(f"{ws(4)}- {exp_name}: \033[93m{executed_num_runs_exp} / {total_num_runs_exp} ({100 * executed_num_runs_exp/total_num_runs_exp} %)\033[0m")


##################################################################################################################################################
# check_experiment_resources
##################################################################################################################################################
##################################################################################################################################################
def check_experiment_baselines_installed(exp_data: Any, exp_yaml: str | Path) -> tuple[int, int, list[str]]:
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment baselines: {exp_yaml}", verb='LOW')
    baselines: dict[str, str] = {}
    for exp_name, settings in exp_data.items():
        baselines[settings.get('Module')] = exp_name

    num_baselines_to_install = 0
    baselines_to_install = []
    for baseline_name, exp_name in baselines.items():
        baseline = get_baseline(baseline_name)
        is_baseline_installed, install_msg = baseline.is_installed()
        if is_baseline_installed:
            print_msg(f"{ws(4)}", f"- {baseline.label}:\033[92m {install_msg}\033[0m", verb='LOW')
        else:    
            print_msg(f"{ws(4)}", f"- {baseline.label}:\033[93m {install_msg}\033[0m", verb='LOW')
            num_baselines_to_install += 1
            baselines_to_install.append(baseline_name)

    num_automatic_install = num_baselines_to_install
    return num_baselines_to_install, num_automatic_install, baselines_to_install

def check_experiment_sequences_available(exp_data: Any, exp_yaml: str | Path) -> tuple[int, int, list[str]]:
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment sequences: {exp_yaml}", verb='LOW')
    
    configs: set[str] = set()
    for _, settings in exp_data.items():
        configs.add(settings.get('Config'))

    sequences : dict[str, str] = {}
    for config_yaml in configs:
        config_file = os.path.join(VSLAM_LAB_DIR, 'configs', config_yaml)
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names: 
                    sequences[sequence_name] = dataset_name
    
    # Check sequence availability
    sequences_to_download = {}
    num_total_sequences = len(sequences)
    num_available_sequences = 0
    for sequence_name, dataset_name in sequences.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        if dataset_name not in sequences_to_download:
            sequences_to_download[dataset_name] = []
        if dataset.check_sequence_availability(sequence_name, False) == "available":
            print_msg(f"{ws(4)}", f"- {dataset.dataset_label} {dataset.dataset_color}{sequence_name}:\033[92m available\033[0m", verb='MEDIUM')
            num_available_sequences += 1
        else:
            print_msg(f"{ws(4)}", f"- {dataset.dataset_label} {dataset.dataset_color}{sequence_name}:\033[93m not available\033[0m \033[92m (automatic download)\033[0m", verb='MEDIUM')
            sequences_to_download[dataset_name].append(sequence_name)
    print_msg(f"{ws(4)}", f"- Sequences available: \033[92m{num_available_sequences}\033[0m / {num_total_sequences}", verb='LOW')
    if num_available_sequences < num_total_sequences:
        print_msg(f"{ws(4)}", f"- Sequences to download: \033[93m{num_total_sequences - num_available_sequences}\033[0m / {num_total_sequences}", verb='LOW')

    # Check download issues
    num_download_issues = 0
    num_automatic_solution = 0
    first_download_issue_found = False
    for dataset_name, sequence_names in sequences_to_download.items():
        if sequence_names == []:
            continue
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        issues_seq = dataset.get_download_issues(sequence_names)
        if issues_seq:
            if not first_download_issue_found:
                print_msg(f"\n{ws(4)}", "LIST OF DOWNLOAD ISSUES:",'warning')
            first_download_issue_found = True
        for issue_seq in issues_seq:
            print_msg(f"{ws(4)}", f"- [{dataset_name}][{issue_seq['name']}]: {issue_seq['description']}",'warning')
            print(f"{ws(8)}[{issue_seq['mode']}]: {issue_seq['solution']}")
            num_download_issues += 1
            if issue_seq['mode'] == '\033[92mautomatic download\033[0m':
                num_automatic_solution += 1

    return num_download_issues, num_automatic_solution, sequences_to_download

def check_experiment_resources(exp_yaml: str | Path) -> tuple[list[str], list[str]]:
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
   
    num_baselines_to_install, num_automatic_install, baselines_to_install = check_experiment_baselines_installed(exp_data, exp_yaml)
    num_download_issues, num_automatic_download, sequences_to_download = check_experiment_sequences_available(exp_data, exp_yaml)
    
    if num_baselines_to_install > 0 or num_download_issues > 0:
        print_msg(f"\n{SCRIPT_LABEL}",f"Your experiments have {num_baselines_to_install} install issues and {num_download_issues} download issues:",'warning')
        if(num_baselines_to_install - num_automatic_install) > 0:
            print_msg(f"{ws(4)}",f"- {num_baselines_to_install - num_automatic_install} baselines need to be installed manually.",'warning')
            print_msg(f"{ws(4)}", f"Some issues are  not automatically fixable. Please, fix them manually and run the experiment again.",'error')
            exit(1)
        if num_download_issues - num_automatic_download > 0:
            print_msg(f"{ws(4)}", f"Some issues are  not automatically fixable. Please, fix them manually and run the experiment again.",'error')
            exit(1)
        
        print(f"{ws(4)}All issues are \033[92mautomatically\033[0m fixable.")

    return baselines_to_install, sequences_to_download, num_download_issues

def get_experiment_resources(exp_yaml: str | Path) -> None:
    baselines_to_install, sequences_to_download, num_download_issues = check_experiment_resources(exp_yaml)
    if num_download_issues > 0:
        #print_msg(f"\n{SCRIPT_LABEL}",f"Your experiments have {num_baselines_to_install} install issues and {num_download_issues} download issues:",'warning')
        message = (
            f"{SCRIPT_LABEL}"
            f"There {'is' if num_download_issues == 1 else 'are'} "
            f"{num_download_issues} download/install issue"
            f"{'' if num_download_issues == 1 else 's'} "
            "\033[92mautomatically\033[0m fixable. "
            "Would you like to continue solving them (Y/n): "
        )
        #message = (f"{SCRIPT_LABEL}There is {num_download_issues} download/install issues \033[92mautomatically\033[0m fixable. Would you like to continue solving them (Y/n): ")
        try:
            user_input = inputimeout(prompt=message, timeout=60*10).strip()
        except TimeoutOccurred:
            user_input = 'Y'
            print(f"{ws(4)}No input detected. Defaulting to 'Y'.")
        if user_input == 'n':
            exit() 

    install_baselines(baselines_to_install)

    first_time = True
    for dataset_name, sequence_names in sequences_to_download.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for sequence_name in sequence_names:
            if first_time:
                print(f"\n{SCRIPT_LABEL}Downloading (to {VSLAMLAB_BENCHMARK}) ...")
                first_time = False
            dataset.download_sequence(sequence_name)

##################################################################################################################################################
# update_experiment_csv_log
##################################################################################################################################################
##################################################################################################################################################
def update_experiment_csv_log(exp_name: str, settings: Any) -> bool:
    updated = False

    exp_folder = VSLAMLAB_EVALUATION / exp_name
    exp_log_csv = exp_folder / "vslamlab_exp_log.csv"
    exp_log = pd.read_csv(exp_log_csv)

    baseline_name = settings.get('Module')
    all_match = exp_log["method_name"].eq(baseline_name).all()
    if not all_match:
        print_msg(f"{SCRIPT_LABEL}", f"The original method cannot be changed ({(exp_log["method_name"][0])} != {baseline_name}). Only new sequences or more runs can be added to the experiment.",'error')
        exit(1)

    config_yaml = settings.get('Config')
    config_file = os.path.join(VSLAM_LAB_DIR, 'configs', config_yaml)
    num_runs = settings.get('NumRuns')
    with open(config_file, 'r') as file:
        config_file_data = yaml.safe_load(file)
        for dataset_name, sequence_names in config_file_data.items():
            for sequence_name in sequence_names: 
                for iRun in range(0, num_runs):
                        subset = exp_log[
                            (exp_log["dataset_name"] == dataset_name) &
                            (exp_log["sequence_name"] == sequence_name) &
                            (exp_log["exp_it"] == iRun)]

                        if subset.empty:
                            updated = True
                            new_row = {
                                "method_name": baseline_name,
                                "dataset_name": dataset_name,
                                "sequence_name": sequence_name,
                                "exp_it": iRun,
                                "STATUS": "",
                                "SUCCESS": "",
                                "TIME": "0.0",
                                "RAM": "0.0",
                                "SWAP": "0.0",
                                "GPU": "0.0",
                                "COMMENTS": "",
                                "EVALUATION": "none",
                                "num_frames": "0",
                                "num_tracked_frames": "0",
                                "num_evaluated_frames": "0",
                            }
                            exp_log = pd.concat(
                                [exp_log, pd.DataFrame([new_row])],
                                ignore_index=True,
                            )
    if updated:
        exp_log.to_csv(exp_log_csv, index=False)
    return updated
         
def create_experiment_csv_log(exp_name: str, settings: Any) -> None:
    exp_folder = VSLAMLAB_EVALUATION / exp_name
    exp_log_csv = exp_folder / "vslamlab_exp_log.csv"
    if not exp_folder.exists():
        exp_folder.mkdir(parents=True, exist_ok=True)

    if exp_log_csv.exists(): 
        return

    log_headers = ["method_name", "dataset_name", "sequence_name", "exp_it", "STATUS", "SUCCESS", "TIME", "RAM", "SWAP", "GPU", "COMMENTS", "EVALUATION"]

    with open(exp_log_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_headers)
        config_yaml = os.path.join(VSLAM_LAB_DIR, "configs", settings.get("Config"))
        num_runs = settings.get("NumRuns")
        baseline_name = settings.get("Module")
        with open(config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for i in range(num_runs):
                for dataset_name, sequence_names in config_file_data.items():
                    for sequence_name in sequence_names:
                        exp_it = str(i).zfill(5)  
                        writer.writerow([baseline_name, dataset_name, sequence_name, f"{exp_it}", "", "",0.0, 0.0, 0.0, 0.0, "", "none"])

def update_experiment_csv_logs(exp_yaml: str | Path) -> None:
    
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    num_updates = 0
    for exp_name, settings in exp_data.items():
        exp_folder = VSLAMLAB_EVALUATION / exp_name
        exp_log_csv = exp_folder / "vslamlab_exp_log.csv"
        if not exp_folder.exists():
            exp_folder.mkdir(parents=True, exist_ok=True)

        if not exp_log_csv.exists(): 
            if num_updates == 0:
                print_msg(f"\n{SCRIPT_LABEL}", f"Update experiment csv logs: {exp_yaml}", verb='LOW')
            print(f"{ws(4)}- \033[92mCreate new\033[0m: {exp_log_csv}")
            create_experiment_csv_log(exp_name, settings)
            num_updates += 1
        elif update_experiment_csv_log(exp_name, settings):
            if num_updates == 0:
                print_msg(f"\n{SCRIPT_LABEL}", f"Update experiment csv logs: {exp_yaml}", verb='LOW')
            print(f"{ws(4)}- \033[92mUpdate\033[0m: {exp_log_csv}")
            num_updates += 1

    if num_updates == 0:
        print_msg(f"\n{SCRIPT_LABEL}", f"Update experiment csv logs: {exp_yaml} : \033[92mEverything up-to-date\033[0m", verb='LOW')

##################################################################################################################################################
# overwrite_exp
##################################################################################################################################################
##################################################################################################################################################
def overwrite_exp(exp_yaml: str | Path) -> None:
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
    print_msg(f"\n{SCRIPT_LABEL}", f"Overwrite experiment: '{exp_yaml}'", "warning")
    for exp_name, _ in exp_data.items():
        exp_folder = VSLAMLAB_EVALUATION / exp_name
        exp_folder.mkdir(parents=True, exist_ok=True)
        print_msg(ws(4), f"- Delete:'{exp_folder}'", "warning")

        for item in exp_folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

##################################################################################################################################################
# validate_experiment_yaml
##################################################################################################################################################
##################################################################################################################################################

###################### Check experiment syntax ######################
def check_experiment_baseline_names(exp_data: Any, exp_yaml: str | Path) -> None:
    errors: list[str] = []
    for exp_name, settings in exp_data.items():
        baseline_name = settings.get("Module")
        baseline = get_baseline(baseline_name)
        if baseline == "Invalid case":
            errors.append(
                f"[Error] Module: '{baseline_name}' baseline in '{exp_name}' doesn't exist."
            )

    if not errors:
        return

    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment baseline names (in '{exp_yaml}'):", "info")
    for error in errors:
        print_msg(ws(4), error, "error")
    print_baselines()
    sys.exit(1)

def check_experiment_sequence_names(exp_data: Any, exp_yaml: str | Path) -> None:
    errors: list[str] = []
    configs: set[str] = set()
    for _, settings in exp_data.items():
        config_yaml = settings.get("Config")
        configs.add(config_yaml)
    
    dataset_list = set(list_available_datasets())
    
    for config_yaml in configs:
        config_file = VSLAM_LAB_DIR / 'configs' / config_yaml
        config_file_data = load_yaml_file(config_file)
        
        for dataset_name, sequence_names in config_file_data.items():
            if dataset_name not in dataset_list:
                errors.append(f"[Error] Dataset '{dataset_name}' doesn't exist (in config '{config_file}').")
                continue

            dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)

            for sequence_name in sequence_names:
                if not dataset.contains_sequence(sequence_name):
                    errors.append(
                        f"[Error] Sequence '{sequence_name}' in dataset '{dataset_name}' doesn't exist (in config '{config_file}')."
                    )

    if not errors:
        return
    
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment dataset and sequence names (in '{exp_yaml}'):", "info")
    for error in errors:
        print_msg(ws(4), error, "error")

    print_datasets()
    sys.exit(1)    

###################### Check experiment conflicts ######################
def check_experiment_baselines_conflicts(exp_data:  Any, exp_yaml: str | Path,) -> str:

    errors: list[str] = []
    modes: list[str] = []

    for exp_name, settings in exp_data.items():
        baseline_name = settings.get("Module")
        baseline = get_baseline(baseline_name)
    
        mode = (settings.get("Parameters", {}).get("mode") or baseline.default_parameters.get("mode"))
        if not mode in modes:
            modes.append(mode)

        if mode not in baseline.modes:
            errors.append(
                f"[Error] Baseline '{baseline_name}' in '{exp_name}' doesn't handle "
                f"mode '{mode}'. Available modes are: {baseline.modes}."
            )
    if len(modes) > 1:
        errors.append(f"[Error] Only one mode is allowed per config file. Conflicts: {modes}")

    if errors:
        print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment baseline conflicts (in '{exp_yaml}'):", "info")
        for error in errors:
            print_msg(ws(4), error, "error")
        sys.exit(1)

    return modes[0]

def check_experiment_sequence_conflicts(exp_data:  Any, exp_yaml: str | Path, config_mode: str) -> None:
    errors: list[str] = []
    configs: set[str] = set()
    for _, settings in exp_data.items():
        config_yaml = Path(settings.get("Config"))
        configs.add(config_yaml)

    for config_yaml in configs:
        config_file = VSLAM_LAB_DIR / 'configs' / config_yaml
        config_file_data = load_yaml_file(config_file)
    
        for dataset_name in config_file_data.keys():
            dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
            if config_mode not in dataset.modes:
                errors.append(
                    f"[Error] Dataset '{dataset_name}' (in config '{config_file}') doesn't handle mode "
                    f"'{config_mode}'. Available modes are: {dataset.modes}."
                )

    if not errors:
        return

    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment dataset conflicts (in '{exp_yaml}'):", "info")
    for error in errors:
        print_msg(ws(4), error, "error")

    sys.exit(1)

def validate_experiment_yaml(exp_yaml: str | Path) -> None:
    # Load experiments
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
   
    # Check syntax
    check_experiment_baseline_names(exp_data, exp_yaml)
    check_experiment_sequence_names(exp_data, exp_yaml)

    # Check conflicts
    config_mode = check_experiment_baselines_conflicts(exp_data, exp_yaml)
    check_experiment_sequence_conflicts(exp_data, exp_yaml, config_mode)

    # Print Summary
    print_msg(f"\n{SCRIPT_LABEL}", f"Experiment summary: {exp_yaml}", flag="info", verb='NONE')
    for exp_name, settings in exp_data.items():
        baseline_name = settings.get("Module")
        config = settings.get("Config")
        numRuns = settings.get("NumRuns")
        print(f"{ws(4)} - {exp_name}: \033[96m{baseline_name}\033[0m, \033[38;2;255;165;0m {config}\033[0m x{numRuns}")            