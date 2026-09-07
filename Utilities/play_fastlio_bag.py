#!/usr/bin/python3
"""Replay Ouster data from a ROS 2 bag with SPARK FAST-LIO compatibility fixes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2, PointField


POINTS_TOPIC = "/ouster/points"
IMU_TOPIC = "/ouster/imu"


class OusterPlayer(Node):
    def __init__(self) -> None:
        super().__init__("vslamlab_fast_lio_bag_player")
        self.points_publisher = self.create_publisher(
            PointCloud2,
            POINTS_TOPIC,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.imu_publisher = self.create_publisher(
            Imu,
            IMU_TOPIC,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

    @staticmethod
    def prepare_points(message: PointCloud2) -> PointCloud2:
        ring = next((field for field in message.fields if field.name == "ring"), None)
        if ring is None:
            raise ValueError("Ouster PointCloud2 lacks the required ring field")
        if ring.datatype == PointField.UINT16:
            # SPARK FAST-LIO's registered Ouster PCL type expects uint8. Rings
            # are 0..127, and the low byte at the existing offset is unchanged.
            ring.datatype = PointField.UINT8
        elif ring.datatype != PointField.UINT8:
            raise ValueError(f"Unsupported Ouster ring datatype: {ring.datatype}")
        return message

    def play(self, bag: Path, rate: float) -> int:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        reader.set_filter(rosbag2_py.StorageFilter(topics=[POINTS_TOPIC, IMU_TOPIC]))
        first_record_ns: int | None = None
        started = time.monotonic() + 2.0
        count = 0
        while reader.has_next() and rclpy.ok():
            topic, serialized, record_ns = reader.read_next()
            if first_record_ns is None:
                first_record_ns = record_ns
            target = started + (record_ns - first_record_ns) / 1e9 / rate
            while rclpy.ok():
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.01))
            if topic == POINTS_TOPIC:
                self.points_publisher.publish(self.prepare_points(deserialize_message(serialized, PointCloud2)))
            elif topic == IMU_TOPIC:
                self.imu_publisher.publish(deserialize_message(serialized, Imu))
            count += 1
            rclpy.spin_once(self, timeout_sec=0.0)
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--rate", type=float, default=1.0)
    args = parser.parse_args()
    if args.rate <= 0:
        raise ValueError("Playback rate must be positive")
    rclpy.init()
    node = OusterPlayer()
    try:
        count = node.play(args.bag.expanduser().resolve(), args.rate)
        if count == 0:
            raise RuntimeError("No Ouster point or IMU messages were found")
        node.get_logger().info(f"Published {count} Ouster messages")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
