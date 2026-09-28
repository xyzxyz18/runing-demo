"""Rule-based gait event detection for fixed-camera side-view video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks


@dataclass
class FootEvents:
    strikes: List[int]
    toe_offs: List[int]


def _enforce_spacing(indices: np.ndarray, scores: np.ndarray, min_frames: int) -> List[int]:
    order = indices[np.argsort(scores[indices])[::-1]]
    selected: List[int] = []
    for idx in order:
        if all(abs(int(idx) - old) >= min_frames for old in selected):
            selected.append(int(idx))
    return sorted(selected)


def detect_foot_events(foot_x: np.ndarray, foot_y: np.ndarray, fps: float,
                       min_interval_seconds: float = 0.22,
                       ground_percentile: float = 72.0,
                       velocity_tolerance: float = 0.42) -> FootEvents:
    """Detect approximate strike/toe-off events from normalized image coordinates.

    Image y increases downward. Strikes are local low points near the runner's
    estimated ground band. Toe-off is the strongest upward transition following
    a strike and before the next strike.
    """
    valid = np.isfinite(foot_y)
    if valid.sum() < max(8, int(fps)):
        return FootEvents([], [])
    y = foot_y.copy()
    x = foot_x.copy()
    ids = np.arange(len(y))
    y[~valid] = np.interp(ids[~valid], ids[valid], y[valid])
    valid_x = np.isfinite(x)
    if valid_x.any():
        x[~valid_x] = np.interp(ids[~valid_x], ids[valid_x], x[valid_x])
    else:
        x[:] = 0.0
    vy = np.gradient(y) * fps
    min_frames = max(2, int(round(min_interval_seconds * fps)))
    ground = float(np.percentile(y, ground_percentile))
    spread = max(float(np.percentile(y, 90) - np.percentile(y, 10)), 1e-4)
    peaks, _ = find_peaks(y, distance=min_frames, prominence=0.035 * spread)
    candidates = peaks[y[peaks] >= ground - 0.18 * spread]
    slow_limit = max(np.percentile(np.abs(vy), 45) * velocity_tolerance, 0.01)
    near_slow = candidates[np.abs(vy[candidates]) <= slow_limit]
    if len(near_slow) >= 2:
        candidates = near_slow
    strikes = _enforce_spacing(candidates, y, min_frames)

    toe_offs: List[int] = []
    for pos, strike in enumerate(strikes):
        end = strikes[pos + 1] if pos + 1 < len(strikes) else min(len(y), strike + int(fps))
        start = strike + max(1, int(0.08 * fps))
        if end - start < 3:
            continue
        # Most negative vy is the clearest upward departure in image coordinates.
        local = start + int(np.argmin(vy[start:end]))
        if vy[local] < -max(0.015, 0.12 * np.percentile(np.abs(vy), 80)):
            toe_offs.append(local)
    return FootEvents(strikes, toe_offs)

