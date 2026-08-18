#!/usr/bin/python3
"""Record the latest LiDAR-corrected FAST-LIO Path as a TUM trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class PathRecorder(Node):
    def __init__(self, output: Path, ready_file: Path, topic: str) -> None:
        super().__init__("vslamlab_fast_lio_path_recorder")
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("", encoding="utf-8")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(RosPath, topic, self._record, qos)
        ready_file.write_text("ready\n", encoding="utf-8")

    def _record(self, message: RosPath) -> None:
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for stamped_pose in message.poses:
                stamp = stamped_pose.header.stamp
                pose = stamped_pose.pose
                timestamp = float(stamp.sec) + float(stamp.nanosec) / 1e9
                stream.write(
                    f"{timestamp:.9f} {pose.position.x:.12e} {pose.position.y:.12e} "
                    f"{pose.position.z:.12e} {pose.orientation.x:.12e} "
                    f"{pose.orientation.y:.12e} {pose.orientation.z:.12e} "
                    f"{pose.orientation.w:.12e}\n"
                )
        temporary.replace(self.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--topic", default="/fast_lio/path")
    args = parser.parse_args()
    rclpy.init()
    node = PathRecorder(args.output.expanduser().resolve(), args.ready_file.expanduser().resolve(), args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
