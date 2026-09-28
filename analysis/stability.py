"""Compare every detected foot-strike cycle against its side's mean path."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pose.mediapipe_pose import LANDMARK_NAMES


SAMPLES_PER_CYCLE = 64
INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}


def _rounded(value: Optional[float]) -> Optional[float]:
    return round(float(value), 3) if value is not None and np.isfinite(value) else None


def foot_cycle_analysis(points: np.ndarray, strikes: Dict[str, List[int]],
                        scale: float, min_visibility: float = 0.45,
                        timestamps: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Return all same-side strike intervals, mean paths and per-cycle RMS errors.

    Both feet use the midpoint of the visible hips as the origin and the same
    body-segment scale. Image Y is inverted so positive Y means upward motion.
    Intervals with too few visible points remain listed but have no path/error.
    """
    if points.ndim != 3 or points.shape[1:] != (33, 4):
        raise ValueError("关键点数组必须为 (帧数, 33, 4)")
    body_scale = float(scale) if np.isfinite(scale) and scale > 1e-6 else 1.0
    phase = np.linspace(0.0, 1.0, SAMPLES_PER_CYCLE)
    hips = np.stack((points[:, INDEX["left_hip"], :],
                     points[:, INDEX["right_hip"], :]), axis=1)
    hip_visible = (np.isfinite(hips[:, :, :2]).all(axis=2) &
                   np.isfinite(hips[:, :, 3]) &
                   (hips[:, :, 3] >= min_visibility))
    pelvis = np.full((len(points), 2), np.nan)
    for frame_id in range(len(points)):
        if hip_visible[frame_id].any():
            pelvis[frame_id] = hips[frame_id, hip_visible[frame_id], :2].mean(axis=0)

    output = {}
    all_deviations = []
    for side in ("left", "right"):
        foot = points[:, INDEX[f"{side}_foot_index"], :]
        relative = (foot[:, :2] - pelvis) / body_scale
        relative[:, 1] *= -1  # Up is positive in the chart.
        visible = (np.isfinite(relative).all(axis=1) &
                   np.isfinite(foot[:, 3]) & (foot[:, 3] >= min_visibility))
        cycles = []
        paths = []
        side_strikes = strikes.get(side, [])
        for number, (start, end) in enumerate(zip(side_strikes, side_strikes[1:]), 1):
            record = {
                "number": number, "start_frame": int(start), "end_frame": int(end),
                "duration_seconds": None, "visibility_percent": 0,
                "path": None, "deviation_body_ratio": None, "status": "无法绘制",
            }
            if 0 <= start < end < len(points):
                if timestamps is not None and len(timestamps) == len(points):
                    duration = float(timestamps[end] - timestamps[start])
                    record["duration_seconds"] = round(duration, 3) if duration >= 0 else None
                indices = np.arange(start, end + 1)
                valid = indices[visible[indices]]
                coverage = len(valid) / len(indices)
                record["visibility_percent"] = round(coverage * 100)
                if len(valid) >= 3:
                    source_phase = (valid - start) / (end - start)
                    if (timestamps is not None and len(timestamps) == len(points) and
                            np.isfinite(timestamps[start:end + 1]).all() and
                            timestamps[end] > timestamps[start] and
                            np.all(np.diff(timestamps[valid]) > 0)):
                        source_phase = ((timestamps[valid] - timestamps[start]) /
                                        (timestamps[end] - timestamps[start]))
                    path = np.column_stack([
                        np.interp(phase, source_phase, relative[valid, axis])
                        for axis in (0, 1)
                    ])
                    record["path"] = np.round(path, 4).tolist()
                    record["status"] = "低置信度" if coverage < 0.6 else "可比较"
                    paths.append(path)
            cycles.append(record)

        mean = np.mean(np.stack(paths), axis=0) if paths else np.empty((0, 2))
        deviations = []
        for cycle in cycles:
            if cycle["path"] is None or len(paths) < 2:
                continue
            path = np.asarray(cycle["path"])
            deviation = float(np.sqrt(np.mean(np.sum((path - mean) ** 2, axis=1))))
            cycle["deviation_body_ratio"] = _rounded(deviation)
            deviations.append(deviation)
            all_deviations.append(deviation)
        side_dispersion = float(np.sqrt(np.mean(np.square(deviations)))) if deviations else None
        output[side] = {
            "cycle_count": len(cycles),
            "drawable_count": len(paths),
            "dispersion_body_ratio": _rounded(side_dispersion),
            "mean_path": np.round(mean, 4).tolist(),
            "cycles": cycles,
        }

    overall = (float(np.sqrt(np.mean(np.square(all_deviations))))
               if all_deviations else None)
    left_mean = np.asarray(output["left"]["mean_path"])
    right_mean = np.asarray(output["right"]["mean_path"])
    side_gap = (float(np.sqrt(np.mean(np.sum((left_mean - right_mean) ** 2, axis=1))))
                if len(left_mean) and len(right_mean) else None)
    if overall is None:
        assessment = "数据不足"
        explanation = "同侧至少需要两个可绘制周期，才能计算周期与平均曲线的差异。"
    elif min(output["left"]["drawable_count"], output["right"]["drawable_count"]) < 3:
        assessment = "样本较少"
        explanation = "已展示全部检测到的周期，但至少一侧不足三个可绘制周期，稳定性结论需谨慎。"
    elif overall < 0.15:
        assessment = "周期较一致"
        explanation = "各周期相对本侧平均曲线的差异较小。"
    elif overall < 0.30:
        assessment = "有一定波动"
        explanation = "各周期相对本侧平均曲线有一定差异。"
    else:
        assessment = "周期波动较大"
        explanation = "部分周期与本侧平均曲线差异较大，建议结合视频检查关键点与着地检测。"
    return {
        "version": 3,
        "left": output["left"], "right": output["right"],
        "overall_dispersion_body_ratio": _rounded(overall),
        "side_mean_gap_body_ratio": _rounded(side_gap),
        "assessment": assessment,
        "explanation": explanation,
        "method": "每侧全部相邻着地事件形成一个周期；足尖相对双髋中点，按身体段长度归一化，按周期相位插值到 64 点。分别计算每个周期与本侧平均曲线的二维均方根距离，再合并为总体偏差。低可见度周期保留显示；少于 3 个可见点的周期列为无法绘制。",
    }
