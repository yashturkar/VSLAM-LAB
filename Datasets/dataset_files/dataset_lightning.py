import os
import yaml
import shutil
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from path_constants import VSLAM_LAB_DIR
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab

class LIGHTNING_dataset(DatasetVSLAMLab):
    """LIGHTNING dataset helper for VSLAMLab benchmark - custom KITTI format."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "lightning") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get raw data path
        self.raw_data_path: str = cfg.get("raw_data_path", "")
        
        # Load calibration from lightning.yaml or lightning_settings.yaml
        # Try multiple possible locations and filenames
        possible_locations = [
            str(VSLAM_LAB_DIR / "lightning.yaml"),
            os.path.join(benchmark_path, "lightning_settings.yaml"),
            os.path.join(benchmark_path, "lightning.yaml"),
            os.path.join(self.raw_data_path, "lightning.yaml"),
            os.path.join(self.raw_data_path, "lightning_settings.yaml"),
        ]
        
        self.lightning_cfg = {}
        for lightning_yaml_path in possible_locations:
            if os.path.exists(lightning_yaml_path):
                with open(lightning_yaml_path, "r", encoding="utf-8") as f:
                    self.lightning_cfg = yaml.safe_load(f) or {}
                break

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:
        # Data already exists at raw_data_path, no download needed
        pass

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path_0 = os.path.join(sequence_path, 'rgb_0')
        if not os.path.exists(rgb_path_0):
            os.makedirs(rgb_path_0)

        # Source path: raw_data_path/sequences/{sequence_name}/image_0/
        source_image_path_0 = os.path.join(self.raw_data_path, sequence_name, 'sequences', sequence_name, 'image_0')
        if os.path.exists(source_image_path_0):
            # Copy images from image_0
            for img_file in os.listdir(source_image_path_0):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    src_file = os.path.join(source_image_path_0, img_file)
                    dst_file = os.path.join(rgb_path_0, img_file)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

        # Also create rgb_1 if stereo mode is enabled
        if 'stereo' in self.modes:
            rgb_path_1 = os.path.join(sequence_path, 'rgb_1')
            if not os.path.exists(rgb_path_1):
                os.makedirs(rgb_path_1)

            source_image_path_1 = os.path.join(self.raw_data_path, sequence_name, 'sequences', sequence_name, 'image_1')
            if os.path.exists(source_image_path_1):
                # Copy images from image_1
                for img_file in os.listdir(source_image_path_1):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        src_file = os.path.join(source_image_path_1, img_file)
                        dst_file = os.path.join(rgb_path_1, img_file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        # Read timestamps from raw data
        times_txt = os.path.join(self.raw_data_path, sequence_name, 'sequences', sequence_name, 'times.txt')
        if not os.path.exists(times_txt):
            return

        # Read timestamps
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Collect and sort image filenames
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()

        # Write NEW format (for pycuvslam, droidslam, etc.)
        # NEW format uses underscores and nanoseconds: ts_rgb_0 (ns), path_rgb_0
        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0'])
            for t, fname in zip(times, rgb_files):
                ts_ns = int(t * 1e9)  # Convert seconds to nanoseconds
                writer.writerow([str(ts_ns), f"rgb_0/{fname}"])
        
        # Write OLD format (for mast3rslam)
        # OLD format uses no underscores and seconds: ts_rgb0 (s), path_rgb0
        rgb_mast3r_csv = os.path.join(sequence_path, 'rgb_mast3r.csv')
        with open(rgb_mast3r_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb0 (s)', 'path_rgb0', 'DEBUG_TAG_DATASET_LIGHTNING'])
            for t, fname in zip(times, rgb_files):
                writer.writerow([f"{t:.6f}", f"rgb_0/{fname}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # Extract calibration from lightning.yaml
        fx = float(self.lightning_cfg.get('Camera.fx', 1446.9127793242951))
        fy = float(self.lightning_cfg.get('Camera.fy', 1451.5846408378259))
        cx = float(self.lightning_cfg.get('Camera.cx', 964.9426652255537))
        cy = float(self.lightning_cfg.get('Camera.cy', 607.0681454495964))
        k1 = float(self.lightning_cfg.get('Camera.k1', -0.13902893236244782))
        k2 = float(self.lightning_cfg.get('Camera.k2', 0.23675668936161912))
        p1 = float(self.lightning_cfg.get('Camera.p1', -0.0006401710568311474))
        p2 = float(self.lightning_cfg.get('Camera.p2', 0.000710816965242213))
        k3 = float(self.lightning_cfg.get('Camera.k3', -0.2731326697949815))
        
        # Get image dimensions from first image in sequence
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        w, h = 1920, 1200  # default
        if os.path.exists(rgb_path):
            import cv2
            rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            if rgb_files:
                img = cv2.imread(os.path.join(rgb_path, rgb_files[0]))
                if img is not None:
                    h, w = img.shape[:2]
        
        # Write OLD OpenCV FileStorage format (YAML:1.0 with Camera0.fx keys)
        # This is required by OLD mast3rslam package
        # 1. NEW Format (YAML 1.2) for droidslam/pycuvslam
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        # cam_mono logic
        yaml_new_lines = ["%YAML 1.2", "---", "cam_mono: rgb_0", "", "cameras:"]
        t_bs_str = "1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0"
        
        yaml_new_lines.append(f"  - {{cam_name: rgb_0,")
        yaml_new_lines.append(f"     cam_type: mono,")
        yaml_new_lines.append(f"     cam_model: OPENCV,")
        yaml_new_lines.append(f"     focal_length: [{fx}, {fy}],")
        yaml_new_lines.append(f"     principal_point: [{cx}, {cy}],")
        yaml_new_lines.append(f"     distortion_coefficients: [{k1}, {k2}, {p1}, {p2}, {k3}],")
        yaml_new_lines.append(f"     image_dimension: [{w}, {h}],")
        yaml_new_lines.append(f"     fps: 5.0,")
        yaml_new_lines.append(f"     T_BS: [{t_bs_str}]")
        yaml_new_lines.append(f"    }}")
        
        with open(calibration_yaml, 'w') as f:
            for line in yaml_new_lines:
                f.write(f"{line}\n")

        # 2. OLD Format (OpenCV FileStorage) for mast3rslam
        calibration_cv_yaml = os.path.join(sequence_path, 'calibration_cv.yaml')
        yaml_content_lines = [
            "%YAML:1.0",
            "---",
            f"Camera0.model: OPENCV",
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
        with open(calibration_cv_yaml, 'w') as f:
            for line in yaml_content_lines:
                f.write(f"{line}\n")

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        out_csv = os.path.join(sequence_path, 'groundtruth.csv')

        # Read timestamps
        times_txt = os.path.join(self.raw_data_path, sequence_name, 'sequences', sequence_name, 'times.txt')
        if not os.path.exists(times_txt):
            return

        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Read trajectory and write CSV
        poses_txt = os.path.join(self.raw_data_path, sequence_name, 'poses', sequence_name + '.txt')
        if not os.path.exists(poses_txt):
            return

        with open(poses_txt, 'r') as src, open(out_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])  # header - seconds

            for idx, line in enumerate(src):
                if idx >= len(times):
                    break  # avoid index error if poses has extra lines
                vals = list(map(float, line.strip().split()))
                # row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                Rm = np.array([[vals[0], vals[1], vals[2]],
                            [vals[4], vals[5], vals[6]],
                            [vals[8], vals[9], vals[10]]], dtype=float)
                tx, ty, tz = vals[3], vals[7], vals[11]
                qx, qy, qz, qw = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
                # Write seconds to groundtruth.csv (Default/OLD format for MASt3R-SLAM)
                ts = times[idx]
                writer.writerow([f"{ts:.6f}", tx, ty, tz, qx, qy, qz, qw])

        # Create additional groundtruth_ns.csv (NEW format for DroidSLAM/PyCuVSLAM)
        out_ns_csv = os.path.join(sequence_path, 'groundtruth_ns.csv')
        with open(poses_txt, 'r') as src, open(out_ns_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
            src.seek(0) # Reset poses file reading
            for idx, line in enumerate(src):
                if idx >= len(times): break
                vals = list(map(float, line.strip().split()))
                # row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                Rm = np.array([[vals[0], vals[1], vals[2]],
                            [vals[4], vals[5], vals[6]],
                            [vals[8], vals[9], vals[10]]], dtype=float)
                tx, ty, tz = vals[3], vals[7], vals[11]
                qx, qy, qz, qw = R.from_matrix(Rm).as_quat()
                
                ts_ns = int(times[idx] * 1e9)
                writer.writerow([str(ts_ns), tx, ty, tz, qx, qy, qz, qw])

    def create_imu_csv(self, sequence_name: str) -> None:
        # No IMU data available
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        imu_csv = os.path.join(sequence_path, 'imu.csv')
        # Create empty IMU CSV with header (format: ts, wx, wy, wz, ax, ay, az)
        with open(imu_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])

    def remove_unused_files(self, sequence_name: str) -> None:
        # No cleanup needed for external dataset
        pass

