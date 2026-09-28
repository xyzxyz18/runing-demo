"""Descriptive repeatability of side-view foot paths across stride cycles."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def foot_cycle_analysis(points: np.ndarray, strikes: Dict[str, List[int]],
                        scale: float, min_visibility: float = 0.45) -> Dict[str, object]:
    """Compare foot positions relative to the hip, normalized by body segment length.

    Each same-side strike-to-strike interval is resampled to 64 phase points.
    Dispersion is RMS distance from the median cycle, in body-scale units.
    This measures observed repeatability, not clinical running quality.
    """
    from pose.mediapipe_pose import LANDMARK_NAMES

    index = {name: i for i, name in enumerate(LANDMARK_NAMES)}
    body_scale = float(scale) if np.isfinite(scale) and scale > 1e-6 else 1.0
    phase = np.linspace(0.0, 1.0, 64)
    output = {}
    for side in ("left", "right"):
        foot = points[:, index[f"{side}_foot_index"], :]
        hip = points[:, index[f"{side}_hip"], :]
        relative = (foot[:, :2] - hip[:, :2]) / body_scale
        visible = (np.isfinite(relative).all(axis=1) &
                   np.isfinite(foot[:, 3]) & np.isfinite(hip[:, 3]) &
                   (foot[:, 3] >= min_visibility) & (hip[:, 3] >= min_visibility))
        cycles = []
        intervals = np.diff(strikes[side])
        reference_interval = float(np.median(intervals)) if len(intervals) else 0.0
        for start, end in zip(strikes[side], strikes[side][1:]):
            if start < 0 or end >= len(points) or end - start < 8:
                continue
            if reference_interval and not 0.75 * reference_interval <= end - start <= 1.30 * reference_interval:
                continue
            indices = np.arange(start, end + 1)
            valid = indices[visible[indices]]
            if len(valid) < max(8, int(len(indices) * 0.7)):
                continue
            source_phase = (valid - start) / (end - start)
            cycle = np.column_stack([
                np.interp(phase, source_phase, relative[valid, axis])
                for axis in (0, 1)
            ])
            cycles.append(cycle)

        if cycles:
            stack = np.stack(cycles)
            median = np.median(stack, axis=0)
            deviation = (float(np.sqrt(np.mean(np.sum((stack - median) ** 2, axis=2))))
                         if len(cycles) >= 2 else None)
            mean_path = np.mean(stack, axis=0)
        else:
            median = np.empty((0, 2))
            mean_path = median
            deviation = None
        output[side] = {
            "cycle_count": len(cycles),
            "dispersion_body_ratio": round(deviation, 3) if deviation is not None else None,
            "mean_path": np.round(mean_path, 3).tolist(),
            "cycles": np.round(np.stack(cycles), 3).tolist() if cycles else [],
        }

    measured = [output[side]["dispersion_body_ratio"] for side in ("left", "right")
                if output[side]["dispersion_body_ratio"] is not None]
    overall = round(float(np.mean(measured)), 3) if measured else None
    if overall is None:
        assessment = "数据不足"
        explanation = "至少需要同侧 2 个有效跨步周期，才能比较足部轨迹的重复性。"
    elif min(output["left"]["cycle_count"], output["right"]["cycle_count"]) < 3:
        assessment = "样本较少"
        explanation = "部分侧别有效跨步少于 3 个，可查看轨迹叠加，但不宜据此判断稳定性。"
    elif overall < 0.15:
        assessment = "轨迹较一致"
        explanation = "本段视频中的足部轨迹周期重合度较高。"
    elif overall < 0.30:
        assessment = "存在一定波动"
        explanation = "本段视频中的足部轨迹存在可见周期差异。"
    else:
        assessment = "轨迹波动较大"
        explanation = "本段视频中的足部轨迹周期差异较大，建议结合画面复核。"
    return {
        "left": output["left"], "right": output["right"],
        "overall_dispersion_body_ratio": overall,
        "assessment": assessment,
        "explanation": explanation,
        "method": "同侧着地间足尖相对髋部二维轨迹；按身体段长度归一化，64 点周期对齐，计算相对中位轨迹的均方根偏差。",
    }
