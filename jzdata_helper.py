#!/usr/bin/env python3
"""
简化版转换脚本：
- 从 KITTI 格式数据集中读取所有 timestamp（文件名）
- 按时间差 time_threshold 分割成多个 scene
- 为每个 sample 生成：
    - timestamp（秒）
    - scene_token
    - prev / next（仅在同一 scene 内连接）
- 输出到 output_root 下的 samples_scenes.json
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict


class JZDataHelper:
    def __init__(self, data_root: str, output_root: str, time_threshold: float = 0.5):
        """
        Args:
            data_root: 数据根目录，至少需要包含 label_2（或任意有时间戳的目录）
            output_root: 输出目录，samples_scenes.json 会保存到这里
            time_threshold: 时间差阈值（秒），超过此值视为新 scene
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.time_threshold = time_threshold

        # 保证输出目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)

    def get_timestamps(self) -> List[Tuple[str, float]]:
        """
        从 label_2 目录读取所有文件名，解析为 (timestamp_str, timestamp_sec) 并排序
        这里默认 timestamp_str 是以纳秒为单位的整数时间戳（文件名不含扩展名）。
        """
        label_dir = self.data_root / "label_2"
        if not label_dir.exists():
            raise ValueError(f"标签目录不存在: {label_dir}")

        file_names = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        timestamps: List[Tuple[str, float]] = []

        for f_name in file_names:
            ts_str = f_name.split(".")[0]
            try:
                ts_ns = float(ts_str)
                ts_sec = ts_ns * 1e-9  # 转为秒
                timestamps.append((ts_str, ts_sec))
            except ValueError:
                print(f"警告: 无法解析时间戳 {ts_str}")

        timestamps.sort(key=lambda x: x[1])
        return timestamps

    def group_into_scenes(self, timestamps: List[Tuple[str, float]]) -> List[List[Tuple[str, float]]]:
        """
        按时间差分 scene：
        - 相邻帧时间差 <= time_threshold 视为同一 scene
        - 否则开启新的 scene
        """
        if not timestamps:
            return []

        scenes: List[List[Tuple[str, float]]] = []
        current_scene: List[Tuple[str, float]] = [timestamps[0]]

        for i in range(1, len(timestamps)):
            prev_t = timestamps[i - 1][1]
            cur_t = timestamps[i][1]
            dt = cur_t - prev_t

            if dt <= self.time_threshold:
                current_scene.append(timestamps[i])
            else:
                print(f"时间差为 {dt:.6f}s, 新建 scene")
                scenes.append(current_scene)
                current_scene = [timestamps[i]]

        if current_scene:
            scenes.append(current_scene)

        return scenes

    # ------------------ 核心逻辑 ------------------ #
    def convert_scene(self, scene_idx: int, scene_timestamps: List[Tuple[str, float]]) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
        """
        将一个 scene 内的所有 timestamp 转成 sample 记录：
        每条记录包含: token(=timestamp_str), timestamp, scene, prev, next, lidar_rel_path
        其中 lidar_rel_path 是相对于 /home/jz/gef/data/2022-02-18/ 的点云相对路径。
        """
        scene_name = f"jz_scene_{scene_idx:04d}"

        samples = {}
        scene_tokens = []

        prev_token = None

        for ts_str, ts_sec in scene_timestamps:
            lidar_rel_path = f"total/velodyne/{ts_str}.bin"
            sample = {
                "timestamp": ts_sec,
                "scene": scene_name,
                "lidar_path": lidar_rel_path,
                "prev": prev_token,
                "next": None,
            }

            if prev_token is not None:
                samples[prev_token]["next"] = ts_str

            samples[ts_str] = sample
            scene_tokens.append(ts_str)
            prev_token = ts_str

        return samples, {scene_name: scene_tokens}

    def convert(self):
        """
        主入口：
        - 读取所有时间戳
        - 按时间差分 scene
        - 为每个 sample 生成 timestamp / scene / prev / next / lidar_rel_path
        - 输出到 samples_scenes.json
        """
        print("开始转换（简化版）...")
        print("读取时间戳...")
        timestamps = self.get_timestamps()
        print(f"共找到 {len(timestamps)} 个时间戳")

        print(f"按时间差阈值 {self.time_threshold}s 分组为 scene ...")
        scenes = self.group_into_scenes(timestamps)
        print(f"共生成 {len(scenes)} 个 scene")
        for i, s in enumerate(scenes):
            print(f"scene_{i}: {len(s)} 个 sample")

        samples_by_token: Dict[str, Dict] = {}
        scenes_index: Dict[str, List[str]] = {}
        for idx, scene_ts in enumerate(scenes):
            scene_samples, scene_map = self.convert_scene(idx, scene_ts)
            samples_by_token.update(scene_samples)
            scenes_index.update(scene_map)

        out_path = self.output_root / "samples_scenes.json"
        out = {
            "samples": samples_by_token,
            "scenes": scenes_index
        }

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n已保存样本信息到: {out_path}")
        print("字段包含: token(=timestamp_str), timestamp, scene, prev, next, lidar_rel_path")


def main():
    data_root = "/home/gef/gef/data/2022-02-18/total"
    # 仅保存场景与时间戳关系的信息，更合理的目录名
    output_root = "/home/gef/gef/data/2022-02-18/scene_info"
    time_threshold = 0.5

    converter = JZDataHelper(data_root, output_root, time_threshold)
    converter.convert()


if __name__ == "__main__":
    main()


