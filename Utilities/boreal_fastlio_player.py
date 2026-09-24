#!/usr/bin/python3
"""Replay BorealHDR lidar/IMU CSV files as ROS 2 messages for FAST-LIO."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField


POINT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "time", "ring"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
    "offsets": [0, 4, 8, 12, 16, 20],
    "itemsize": 24,
})


def stamp(message, nanoseconds: int) -> None:
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(nanoseconds, 1_000_000_000)


def load_cloud(path: Path) -> tuple[int, int, bytes, int]:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if values.shape[1] != 6 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid lidar CSV: {path}")
    rings = values[:, 4]
    if np.any(rings != np.floor(rings)) or np.any((rings < 0) | (rings >= 16)):
        raise ValueError(f"Invalid VLP-16 ring in {path}")
    offsets = values[:, 5]
    minimum, maximum = float(np.min(offsets)), float(np.max(offsets))
    duration = maximum - minimum
    if not 0.07 <= duration <= 0.13:
        raise ValueError(f"Unexpected scan duration {duration:.6f}s: {path}")
    points = np.zeros(len(values), dtype=POINT_DTYPE)
    for column in ("x", "y", "z", "intensity"):
        points[column] = values[:, ("x", "y", "z", "intensity").index(column)]
    points["ring"] = rings.astype(np.uint16)
    points["time"] = (offsets - minimum).astype(np.float32)
    # FAST-LIO uses the last point's offset to determine scan end.
    order = np.argsort(points["time"], kind="stable")
    points = points[order]
    file_ns = int(path.stem)
    return file_ns + round(minimum * 1e9), file_ns + round(maximum * 1e9), points.tobytes(), len(points)


class BorealPlayer(Node):
    def __init__(self) -> None:
        super().__init__("vslamlab_boreal_fastlio_player")
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/boreal/lidar", QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.imu_pub = self.create_publisher(
            Imu, "/boreal/imu", QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))

    def wait_for_mapper(self) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and rclpy.ok():
            if self.cloud_pub.get_subscription_count() and self.imu_pub.get_subscription_count():
                time.sleep(1)
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("FAST-LIO did not subscribe to both Boreal topics")

    def publish_imu(self, row: dict[str, str]) -> None:
        message = Imu()
        stamp(message, int(row["timestamp"]))
        message.header.frame_id = "imu_link"
        message.orientation_covariance[0] = -1.0
        message.angular_velocity.x = float(row["gyro_x"])
        message.angular_velocity.y = float(row["gyro_y"])
        message.angular_velocity.z = float(row["gyro_z"])
        message.linear_acceleration.x = float(row["acc_x"])
        message.linear_acceleration.y = float(row["acc_y"])
        message.linear_acceleration.z = float(row["acc_z"])
        self.imu_pub.publish(message)

    def publish_cloud(self, path: Path) -> tuple[int, int]:
        start_ns, end_ns, data, count = load_cloud(path)
        message = PointCloud2()
        stamp(message, start_ns)
        message.header.frame_id = "lidar_link"
        message.height = 1
        message.width = count
        message.is_bigendian = False
        message.is_dense = True
        message.fields = [
            PointField(name=name, offset=offset, datatype=datatype, count=1)
            for name, offset, datatype in (
                ("x", 0, PointField.FLOAT32), ("y", 4, PointField.FLOAT32),
                ("z", 8, PointField.FLOAT32), ("intensity", 12, PointField.FLOAT32),
                ("time", 16, PointField.FLOAT32), ("ring", 20, PointField.UINT16),
            )
        ]
        message.point_step = POINT_DTYPE.itemsize
        message.row_step = count * message.point_step
        message.data = data
        self.cloud_pub.publish(message)
        return start_ns, end_ns

    def play(self, source: Path, duration_s: float, rate: float) -> None:
        scans = sorted((source / "lidar").glob("*.csv"), key=lambda path: int(path.stem))
        if not scans:
            raise ValueError(f"No lidar scans under {source}")
        imu_path = source / "imu/imu.csv"
        with imu_path.open(newline="", encoding="utf-8") as stream:
            imu = list(csv.DictReader(stream))
        if not imu:
            raise ValueError(f"No IMU samples in {imu_path}")
        start_ns = min(int(imu[0]["timestamp"]), int(scans[0].stem) - 110_000_000)
        end_ns = start_ns + round(duration_s * 1e9) if duration_s else None
        self.wait_for_mapper()
        wall_start = time.monotonic() + 1
        imu_index = scan_index = 0
        while rclpy.ok() and (imu_index < len(imu) or scan_index < len(scans)):
            imu_ns = int(imu[imu_index]["timestamp"]) if imu_index < len(imu) else 2**63
            # Publish scans after their final point has been measured. The file
            # stamp comes from the original Velodyne header, near scan end.
            scan_ns = int(scans[scan_index].stem) + 5_000_000 if scan_index < len(scans) else 2**63
            next_ns = min(imu_ns, scan_ns)
            if end_ns is not None and next_ns > end_ns:
                break
            target = wall_start + (next_ns - start_ns) / 1e9 / rate
            while rclpy.ok() and time.monotonic() < target:
                time.sleep(min(0.01, target - time.monotonic()))
            if imu_ns <= scan_ns:
                self.publish_imu(imu[imu_index])
                imu_index += 1
            else:
                scan_start, scan_end = self.publish_cloud(scans[scan_index])
                if scan_index % 50 == 0:
                    print(f"scan {scan_index + 1}/{len(scans)}: {scan_start}..{scan_end}", flush=True)
                scan_index += 1
        print(f"Published {scan_index} lidar scans and {imu_index} IMU messages", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--duration", type=float, default=0, help="Sensor seconds; zero means full sequence")
    parser.add_argument("--rate", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration < 0 or args.rate <= 0:
        parser.error("duration must be nonnegative and rate must be positive")
    rclpy.init()
    node = BorealPlayer()
    try:
        node.play(args.source.resolve(), args.duration, args.rate)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
