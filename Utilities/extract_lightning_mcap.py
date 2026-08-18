# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcap>=1.3",
#   "mcap-ros2-support>=0.5",
#   "numpy>=2.0",
#   "opencv-python-headless>=4.10",
# ]
# ///
"""Extract a synchronized stereo MCAP recording into a VSLAM-LAB sequence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def timestamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def read_camera_info(bag_path: Path, topics: tuple[str, str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    with bag_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages(topics=list(topics)):
            found.setdefault(channel.topic, message)
            if len(found) == len(topics):
                break
    missing = set(topics) - found.keys()
    if missing:
        raise RuntimeError(f"Missing camera-info topic(s): {sorted(missing)}")
    return found


def camera_values(message: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    size = (int(message.width), int(message.height))
    camera = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(message.d, dtype=np.float64)
    rectification = np.asarray(message.r, dtype=np.float64).reshape(3, 3)
    projection = np.asarray(message.p, dtype=np.float64).reshape(3, 4)
    return camera, distortion, rectification, projection, size


def write_calibration(path: Path, projection: np.ndarray, baseline: float, fps: float, size: tuple[int, int]) -> None:
    fx, fy = float(projection[0, 0]), float(projection[1, 1])
    cx, cy = float(projection[0, 2]), float(projection[1, 2])
    width, height = size
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    right_pose = identity.copy()
    right_pose[3] = baseline

    def camera_block(name: str, pose: list[float]) -> dict[str, Any]:
        return {
            "cam_name": name,
            "cam_type": "mono",
            "cam_model": "radtan5",
            "distortion_type": "radtan5",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "distortion_coefficients": [0.0] * 5,
            "image_dimension": [width, height],
            "fps": fps,
            "T_BS": pose,
        }

    # JSON is valid YAML and avoids requiring PyYAML in the standalone extractor.
    path.write_text(json.dumps({"cameras": [camera_block("rgb_0", identity), camera_block("rgb_1", right_pose)]}, indent=2) + "\n")


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def extract_sequence(
    bag_path: Path,
    output: Path,
    *,
    left_topic: str = "/cam_sync/cam0/image_preview/compressed",
    right_topic: str = "/cam_sync/cam1/image_preview/compressed",
    left_info_topic: str = "/cam_sync/cam0/camera_info",
    right_info_topic: str = "/cam_sync/cam1/camera_info",
    odometry_topic: str = "/odometry",
    expected_pairs: int | None = None,
    overwrite_images: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    source_fingerprint: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract one processed LIGHTNING MCAP and return its metadata."""
    bag_path = bag_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    left_folder, right_folder = output / "rgb_0", output / "rgb_1"
    left_folder.mkdir(parents=True, exist_ok=True)
    right_folder.mkdir(parents=True, exist_ok=True)

    info_topics = (left_info_topic, right_info_topic)
    info = read_camera_info(bag_path, info_topics)
    left_camera, left_distortion, left_rectification, left_projection, left_size = camera_values(info[info_topics[0]])
    right_camera, right_distortion, right_rectification, right_projection, right_size = camera_values(info[info_topics[1]])
    if left_size != right_size:
        raise ValueError(f"Camera image dimensions differ: {left_size} != {right_size}")
    if not np.allclose(left_projection[:, :3], right_projection[:, :3], rtol=0, atol=1e-8):
        raise ValueError("Left/right rectified camera matrices differ")
    rectified_camera = left_projection[:, :3]
    baseline = -float(right_projection[0, 3]) / float(right_projection[0, 0])
    left_maps = cv2.initUndistortRectifyMap(
        left_camera, left_distortion, left_rectification, rectified_camera, left_size, cv2.CV_16SC2
    )
    right_maps = cv2.initUndistortRectifyMap(
        right_camera, right_distortion, right_rectification, rectified_camera, right_size, cv2.CV_16SC2
    )

    image_topics = {left_topic: (left_folder, left_maps), right_topic: (right_folder, right_maps)}
    image_paths: dict[str, dict[int, str]] = {topic: {} for topic in image_topics}
    odometry: dict[int, list[Any]] = {}
    topics = [*image_topics, odometry_topic]
    started = time.monotonic()
    last_reported = 0
    with bag_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            stamp = timestamp_ns(message)
            if topic == odometry_topic:
                pose = message.pose.pose
                odometry.setdefault(
                    stamp,
                    [
                        stamp,
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                )
                continue
            if stamp in image_paths[topic]:
                continue
            folder, maps = image_topics[topic]
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Could not decode {topic} at {stamp}")
            if (image.shape[1], image.shape[0]) != left_size:
                raise ValueError(f"Unexpected image size at {stamp}: {image.shape[1]}x{image.shape[0]}")
            rectified = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
            destination = folder / f"{stamp}.png"
            if (overwrite_images or not destination.exists()) and not cv2.imwrite(str(destination), rectified):
                raise RuntimeError(f"Could not write {destination}")
            image_paths[topic][stamp] = f"{folder.name}/{destination.name}"
            count = len(image_paths[topic])
            if count % 100 == 0:
                print(f"{topic}: {count} unique frames", flush=True)
            paired_so_far = len(image_paths[left_topic].keys() & image_paths[right_topic].keys())
            if progress is not None and paired_so_far >= last_reported + 25:
                elapsed = max(time.monotonic() - started, 1e-9)
                progress({
                    "frames": paired_so_far,
                    "total": expected_pairs,
                    "rate_fps": paired_so_far / elapsed,
                    "elapsed_s": elapsed,
                })
                last_reported = paired_so_far

    left, right = image_paths[left_topic], image_paths[right_topic]
    paired_stamps = sorted(left.keys() & right.keys())
    if not paired_stamps:
        raise RuntimeError("No exactly synchronized stereo frames found")
    if left.keys() != right.keys():
        print(f"Warning: using {len(paired_stamps)} pairs; left={len(left)}, right={len(right)}")
    intervals = np.diff(np.asarray(paired_stamps, dtype=np.int64)) / 1e9
    fps = float(1.0 / np.median(intervals[intervals > 0]))
    write_csv(
        output / "rgb.csv",
        ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"],
        [[stamp, left[stamp], stamp, right[stamp]] for stamp in paired_stamps],
    )
    write_csv(
        output / "groundtruth.csv",
        ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"],
        [odometry[stamp] for stamp in sorted(odometry)],
    )
    write_calibration(output / "calibration.yaml", left_projection, baseline, fps, left_size)
    metadata = {
        "schema_version": 2,
        "status": "complete",
        "source_bag": str(bag_path),
        "source_fingerprint": source_fingerprint,
        "left_topic": left_topic,
        "right_topic": right_topic,
        "odometry_topic": odometry_topic,
        "groundtruth_frame": "odom -> body (camera extrinsic unavailable in bag TF)",
        "stereo_pairs": len(paired_stamps),
        "odometry_poses": len(odometry),
        "fps": fps,
        "rectified_projection_left": list(map(float, left_projection.reshape(-1))),
        "rectified_projection_right": list(map(float, right_projection.reshape(-1))),
        "baseline_m": baseline,
    }
    (output / "extraction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if progress is not None:
        elapsed = max(time.monotonic() - started, 1e-9)
        progress({"frames": len(paired_stamps), "total": len(paired_stamps), "rate_fps": len(paired_stamps) / elapsed, "elapsed_s": elapsed})
    print(f"Prepared {output}: {len(paired_stamps)} stereo pairs, {len(odometry)} GT poses, {fps:.6f} Hz")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path, help="MCAP file")
    parser.add_argument("--output", required=True, type=Path, help="VSLAM-LAB sequence directory")
    parser.add_argument("--left-topic", default="/cam_sync/cam0/image_preview/compressed")
    parser.add_argument("--right-topic", default="/cam_sync/cam1/image_preview/compressed")
    parser.add_argument("--left-info-topic", default="/cam_sync/cam0/camera_info")
    parser.add_argument("--right-info-topic", default="/cam_sync/cam1/camera_info")
    parser.add_argument("--odometry-topic", default="/odometry")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted images")
    args = parser.parse_args()
    extract_sequence(
        args.bag,
        args.output,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        left_info_topic=args.left_info_topic,
        right_info_topic=args.right_info_topic,
        odometry_topic=args.odometry_topic,
        overwrite_images=args.force,
    )


if __name__ == "__main__":
    main()
