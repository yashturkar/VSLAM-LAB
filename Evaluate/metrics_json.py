import os
import json
import subprocess
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from Evaluate.evo_functions import evo_metric, compute_trajectory_length
from path_constants import TRAJECTORY_FILE_NAME, GROUNTRUTH_FILE
from utilities import read_trajectory_csv, read_trajectory_txt

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def compute_rotation_rmse(groundtruth_csv, trajectory_csv, evaluation_folder, max_time_difference=0.1):
    """
    Compute rotation RMSE using evo_rpe (Relative Pose Error).
    
    Args:
        groundtruth_csv: Path to ground truth CSV file
        trajectory_csv: Path to trajectory CSV file
        evaluation_folder: Folder where evaluation results are stored
        max_time_difference: Maximum time difference for trajectory association
        
    Returns:
        dict with rotation RMSE metrics or None if computation fails
    """
    # Use evo_rpe to compute relative pose error
    traj_file_name = os.path.basename(trajectory_csv).replace(".csv", "")
    rpe_zip = os.path.join(evaluation_folder, f"{traj_file_name}_rpe.zip")
    rpe_csv = os.path.join(evaluation_folder, f"{traj_file_name}_rpe.csv")
    
    # Run evo_rpe
    traj_txt = os.path.join(evaluation_folder, f"{traj_file_name}.txt")
    gt_txt = os.path.join(evaluation_folder, "groundtruth.txt")
    
    if not os.path.exists(traj_txt) or not os.path.exists(gt_txt):
        return None
    
    # Run evo_rpe with rotation error
    command = f"evo_rpe tum {gt_txt} {traj_txt} --delta 1 --delta_unit m -va -as --save_results {rpe_zip}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if not os.path.exists(rpe_zip):
        return None
    
    # Use evo_res to parse the results (similar to evo_get_accuracy)
    try:
        command_res = f"pixi run -e vslamlab evo_res {rpe_zip} --save_table {rpe_csv}"
        process_res = subprocess.Popen(command_res, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process_res.communicate()
        
        if os.path.exists(rpe_csv):
            df = pd.read_csv(rpe_csv)
            # evo_rpe outputs rotation error - look for rotation-related columns
            # The column names might vary, but typically include 'rot' or 'rotation'
            rotation_rmse = None
            
            # Try common column names for rotation error
            for col in df.columns:
                if 'rot' in col.lower() and 'rmse' in col.lower():
                    rotation_rmse = float(df[col].iloc[0])
                    break
            
            # If not found, try to find any rotation-related metric
            if rotation_rmse is None:
                for col in df.columns:
                    if 'rot' in col.lower() and ('error' in col.lower() or 'rmse' in col.lower()):
                        rotation_rmse = float(df[col].iloc[0])
                        break
            
            # Clean up
            try:
                os.remove(rpe_zip)
                if os.path.exists(rpe_csv):
                    os.remove(rpe_csv)
            except:
                pass
            
            if rotation_rmse is not None:
                # Convert from degrees to radians if needed (check if value > 1, likely degrees)
                if rotation_rmse > 1.0:
                    rotation_rmse = np.deg2rad(rotation_rmse)
                return {"rmse": rotation_rmse}
    except Exception as e:
        # Clean up on error
        try:
            if os.path.exists(rpe_zip):
                os.remove(rpe_zip)
            if os.path.exists(rpe_csv):
                os.remove(rpe_csv)
        except:
            pass
        return None
    
    return None


def extract_metrics_from_evaluation(ate_csv_path, exp_it):
    """
    Extract ATE metrics from ate.csv file.
    
    Args:
        ate_csv_path: Path to ate.csv file
        exp_it: Experiment iteration (e.g., "00000")
        
    Returns:
        dict with ATE metrics or None if not found
    """
    if not os.path.exists(ate_csv_path):
        return None
    
    try:
        df = pd.read_csv(ate_csv_path)
        traj_name = f"{exp_it}_{TRAJECTORY_FILE_NAME}.txt"
        
        # Find matching row
        matching_rows = df[df['traj_name'] == traj_name]
        if matching_rows.empty:
            # Try with .csv extension
            traj_name_csv = f"{exp_it}_{TRAJECTORY_FILE_NAME}.csv"
            matching_rows = df[df['traj_name'].str.contains(exp_it, na=False)]
        
        if matching_rows.empty:
            # Try to get first row if only one trajectory
            if len(df) == 1:
                matching_rows = df.iloc[[0]]
            else:
                return None
        
        row = matching_rows.iloc[0]
        
        return {
            "mean": float(row.get('mean', 0.0)),
            "std": float(row.get('std', 0.0)),
            "rmse": float(row.get('rmse', 0.0)),
            "max": float(row.get('max', 0.0))
        }
    except Exception as e:
        return None


def compute_trajectory_metrics(evaluation_folder, exp_it, groundtruth_csv):
    """
    Compute trajectory length and length ratio.
    
    Args:
        evaluation_folder: Folder where evaluation results are stored
        exp_it: Experiment iteration (e.g., "00000")
        groundtruth_csv: Path to ground truth CSV file
        
    Returns:
        dict with trajectory_length and length_ratio
    """
    traj_name = f"{exp_it}_{TRAJECTORY_FILE_NAME}.tum"
    traj_tum = os.path.join(evaluation_folder, traj_name)
    
    trajectory_length = None
    gt_length = None
    length_ratio = None
    
    # Compute predicted trajectory length
    if os.path.exists(traj_tum):
        trajectory_length = compute_trajectory_length(traj_tum)
    
    # Compute ground truth trajectory length
    if os.path.exists(groundtruth_csv):
        # Convert ground truth CSV to TUM format temporarily
        gt_df = read_trajectory_csv(groundtruth_csv)
        if gt_df is not None and not gt_df.empty:
            gt_tum = os.path.join(evaluation_folder, "groundtruth_temp.tum")
            try:
                gt_numeric = gt_df.apply(pd.to_numeric, errors='coerce')
                np.savetxt(gt_tum, gt_numeric.values, fmt='%.6f', delimiter=' ', newline='\n')
                gt_length = compute_trajectory_length(gt_tum)
                os.remove(gt_tum)
            except:
                pass
    
    # Compute length ratio
    if trajectory_length is not None and gt_length is not None and gt_length > 0:
        length_ratio = trajectory_length / gt_length
    
    result = {}
    if trajectory_length is not None:
        result["trajectory_length"] = float(trajectory_length)
    if length_ratio is not None:
        result["length_ratio"] = float(length_ratio)
    
    return result if result else None


def generate_metrics_json(exp, dataset, sequence_name, exp_it, status="SUCCESS"):
    """
    Generate metrics.json file in the evaluation folder.
    
    Args:
        exp: Experiment object
        dataset: Dataset object
        sequence_name: Name of the sequence
        exp_it: Experiment iteration (e.g., "00000")
        status: Status of the SLAM run ("SUCCESS" or "FAILURE")
        
    Returns:
        Path to generated metrics.json file or None if generation fails
    """
    evaluation_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name, 'vslamlab_evaluation')
    os.makedirs(evaluation_folder, exist_ok=True)
    
    metrics_json_path = os.path.join(evaluation_folder, "metrics.json")
    
    # Check if trajectory exists
    trajectories_path = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    trajectory_csv = os.path.join(trajectories_path, f"{exp_it}_{TRAJECTORY_FILE_NAME}.csv")
    trajectory_txt = os.path.join(trajectories_path, f"{exp_it}_{TRAJECTORY_FILE_NAME}.txt")
    trajectory_exists = os.path.exists(trajectory_csv) or os.path.exists(trajectory_txt)
    
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
        metrics["rmse"]["total"] = ate_metrics["rmse"]  # Use translation RMSE as total
        # If we have ATE metrics, the run was successful
        metrics["status"] = "SUCCESS"
    
    # Compute rotation RMSE
    groundtruth_csv = Path(exp.folder) / dataset.dataset_folder / sequence_name / GROUNTRUTH_FILE
    
    # Use CSV if available, otherwise try TXT
    if not os.path.exists(trajectory_csv):
        trajectory_csv = trajectory_txt
    
    if os.path.exists(trajectory_csv) and os.path.exists(groundtruth_csv):
        rotation_metrics = compute_rotation_rmse(
            groundtruth_csv, 
            trajectory_csv, 
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
    
    # Final check: if we have any meaningful metrics, mark as SUCCESS
    if metrics["ate"] is not None or metrics["rmse"]["translation"] is not None:
        metrics["status"] = "SUCCESS"
    elif metrics["trajectory_length"] is not None:
        # Even if no ATE, if we have trajectory length, it's a partial success
        metrics["status"] = "SUCCESS"
    
    # Write metrics.json
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_json_path

