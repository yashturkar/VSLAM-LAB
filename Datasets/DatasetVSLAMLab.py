"""
Module: VSLAM-LAB - Datasets - DatasetVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2024-07-12
- Updated: 2025-12-30
- License: GPLv3 License

DatasetVSLAMLab: A class to handle Visual SLAM dataset-related operations.

"""

import sys
import yaml
from loguru import logger
from pathlib import Path
from typing import List, Union
from abc import ABC, abstractmethod

from utilities import ws, print_msg
from path_constants import VSLAM_LAB_DIR
from Datasets.DatasetVSLAMLab_calibration import (
    _get_rgb_yaml_section,
    _get_imu_yaml_section,
    _get_rgbd_yaml_section
)

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class DatasetVSLAMLab(ABC):
    """Base dataset class for VSLAM-LAB."""

    # ---- Abstract hooks that concrete datasets must implement ----
    @abstractmethod
    def __init__(self, dataset_name: str, benchmark_path: Union[str, Path]) -> None:  
        # Basic fields
        self.dataset_name: str = dataset_name
        self.dataset_color: str = "\033[38;2;255;165;0m"
        self.dataset_label: str = f"{self.dataset_color}{dataset_name}\033[0m"
        self.dataset_folder: str = dataset_name.upper()

        # Paths
        self.benchmark_path: Path = Path(benchmark_path)
        self.dataset_path: Path = self.benchmark_path / self.dataset_folder
        self.yaml_file: Path = VSLAM_LAB_DIR / "Datasets" / "dataset_files" / f"dataset_{self.dataset_name}.yaml"

        # Load YAML config
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.sequence_names: List[str] = cfg["sequence_names"]
        self.rgb_hz: float = float(cfg["rgb_hz"])
        self.modes: List[str] = cfg.get("modes", ["mono"])
        self.sequence_nicknames: List[str] = []
        self.cam_models: List[str] = cfg.get("cam_models", ["pinhole"])
        
    @abstractmethod
    def download_sequence_data(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_rgb_folder(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_rgb_csv(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_calibration_yaml(self, sequence_name: str) -> None: ...

    def create_imu_csv(self, sequence_name: str) -> None:
        pass
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        pass
    def remove_unused_files(self, sequence_name: str) -> None: 
        pass
    def get_download_issues(self, sequence_names: List[str]) -> dict:
        return {}
    
    ####################################################################################################################
    # Download methods
    def download_sequence(self, sequence_name: str) -> None:

        # Check if sequence is already available
        sequence_availability = self.check_sequence_availability(sequence_name, verbose=True)
        if sequence_availability == "available":
            #print(f"{SCRIPT_LABEL}Sequence {self.dataset_color}{sequence_name}:\033[92m downloaded\033[0m")
            return
        if sequence_availability == "corrupted":
            logger.error(f"\n{ws(4)}Files in sequence {sequence_name} are corrupted.\n{ws(4)}Removing and downloading again sequence {sequence_name}.\n{ws(4)}THIS PART OF THE CODE IS NOT YET IMPLEMENTED. REMOVE THE FILES MANUALLY ")
            sys.exit(1)

        # Download process
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.download_process(sequence_name)

    def download_process(self, sequence_name: str) -> None:
        msg = f"Downloading sequence {self.dataset_color}{sequence_name}\033[0m from dataset {self.dataset_color}{self.dataset_name}\033[0m ..."
        print_msg(SCRIPT_LABEL, msg)
        self.download_sequence_data(sequence_name)
        self.create_rgb_folder(sequence_name)
        self.create_rgb_csv(sequence_name)
        self.create_imu_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)
        self.create_groundtruth_csv(sequence_name)
        self.remove_unused_files(sequence_name)

    ####################################################################################################################
    # Auxiliary methods
    def write_calibration_yaml(self, sequence_name: str, rgb=None, rgbd=None, imu=None) -> None:
        sequence_path = self.dataset_path / sequence_name
        calibration_yaml = sequence_path / 'calibration.yaml'
        
        # Get camera parameters from first rgb camera
        if rgb and len(rgb) > 0:
            cam = rgb[0]
            
            # Handle both formats: individual keys (fx, fy) and list format (focal_length)
            focal = cam.get('focal_length', [1446.91, 1451.58])
            fx = cam.get('fx', focal[0] if isinstance(focal, list) else 1446.91)
            fy = cam.get('fy', focal[1] if isinstance(focal, list) and len(focal) > 1 else 1451.58)
            
            pp = cam.get('principal_point', [964.94, 607.07])
            cx = cam.get('cx', pp[0] if isinstance(pp, list) else 964.94)
            cy = cam.get('cy', pp[1] if isinstance(pp, list) and len(pp) > 1 else 607.07)
            
            # Handle distortion_coefficients list or individual keys
            dist = cam.get('distortion_coefficients', [0.0, 0.0, 0.0, 0.0, 0.0])
            k1 = cam.get('k1', dist[0] if isinstance(dist, list) else 0.0)
            k2 = cam.get('k2', dist[1] if isinstance(dist, list) and len(dist) > 1 else 0.0)
            p1 = cam.get('p1', dist[2] if isinstance(dist, list) and len(dist) > 2 else 0.0)
            p2 = cam.get('p2', dist[3] if isinstance(dist, list) and len(dist) > 3 else 0.0)
            k3 = cam.get('k3', dist[4] if isinstance(dist, list) and len(dist) > 4 else 0.0)
            
            model = cam.get('model', cam.get('cam_model', 'OPENCV'))
            
            # Get image dimensions from first image
            w, h = 1920, 1200  # default
            rgb_0_path = sequence_path / 'rgb_0'
            if rgb_0_path.exists():
                rgb_files = sorted([f for f in rgb_0_path.iterdir() 
                                  if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                if rgb_files:
                    try:
                        import cv2
                        img = cv2.imread(str(rgb_files[0]))
                        if img is not None:
                            h, w = img.shape[:2]
                    except Exception:
                        pass
            
            # Write OLD OpenCV FileStorage format (YAML:1.0 with Camera0.fx keys)
            # This is required by OLD mast3rslam package
            yaml_content_lines = [
                "%YAML:1.0",
                "---",
                f"Camera0.model: {model}",
                f"Camera0.fx: {fx}",
                f"Camera0.fy: {fy}",
                f"Camera0.cx: {cx}",
                f"Camera0.cy: {cy}",
                f"Camera0.k1: {k1}",
                f"Camera0.k2: {k2}",
                f"Camera0.p1: {p1}",
                f"Camera0.p2: {p2}",
                f"Camera0.k3: {k3}",
                f"Camera0.w: {w}",
                f"Camera0.h: {h}",
            ]
        else:
            # Fallback if no rgb camera provided
            yaml_content_lines = [
                "%YAML:1.0",
                "---",
                "Camera0.model: OPENCV",
                "Camera0.fx: 1446.91",
                "Camera0.fy: 1451.58",
                "Camera0.cx: 964.94",
                "Camera0.cy: 607.07",
                "Camera0.k1: 0.0",
                "Camera0.k2: 0.0",
                "Camera0.p1: 0.0",
                "Camera0.p2: 0.0",
                "Camera0.k3: 0.0",
                "Camera0.w: 1920",
                "Camera0.h: 1200",
            ]
        
        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def check_sequence_availability(self, sequence_name: str, verbose: bool = True) -> str:
        sequence_path = self.dataset_path / sequence_name
        if sequence_path.is_dir():
            sequence_complete = self.check_sequence_integrity(sequence_name, verbose=verbose)
            if sequence_complete:
                return "available"
            else:
                return "corrupted"
        return "non-available"

    def check_sequence_integrity(self, sequence_name: str, verbose: bool) -> bool:
        sequence_path = self.dataset_path / sequence_name

        # Define requirements: (Path, Description, is_directory)
        requirements = [
            (sequence_path, "Sequence folder", True),
            (sequence_path / 'rgb_0', "RGB folder", True),
            (sequence_path / 'rgb.csv', "RGB timestamp CSV", False),
            (sequence_path / 'calibration.yaml', "Calibration YAML", False),
        ]
        if 'stereo' in self.modes:
            requirements.append((sequence_path / 'rgb_1', "Right RGB folder", True))
        if 'mono-vi' in self.modes:
            requirements.append((sequence_path / 'imu_0.csv', "IMU CSV", False))

        # Check all requirements
        complete_sequence = True
        for path_obj, desc, should_be_dir in requirements:
            exists = path_obj.is_dir() if should_be_dir else path_obj.is_file()
            if not exists:
                if verbose:
                    logger.error(f"\n{ws(4)}Missing {desc}: {path_obj} !!!!!")
                complete_sequence = False

        return complete_sequence

    ####################################################################################################################
    # Utils

    def contains_sequence(self, sequence_name_ref: str) -> bool:
        return sequence_name_ref in self.sequence_names

    def print_sequence_names(self) -> None:
        print(self.sequence_names)

    def print_sequence_nicknames(self) -> None:
        print(self.sequence_nicknames)

    def get_sequence_names(self) -> list:
        return self.sequence_names

    def get_sequence_nicknames(self) -> list:
        return self.sequence_nicknames

    def get_sequence_nickname(self, sequence_name_ref: str) -> str:
        idx = self.sequence_names.index(sequence_name_ref)
        return self.sequence_nicknames[idx]