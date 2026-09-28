"""Foot position and normalized velocity helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def body_scale(shoulder: np.ndarray, hip: np.ndarray, knee: np.ndarray,
               ankle: np.ndarray) -> float:
    segments = np.concatenate([
        np.linalg.norm(shoulder[:, :2] - hip[:, :2], axis=1),
        np.linalg.norm(hip[:, :2] - knee[:, :2], axis=1),
        np.linalg.norm(knee[:, :2] - ankle[:, :2], axis=1),
    ])
    finite = segments[np.isfinite(segments) & (segments > 1e-5)]
    return float(np.median(finite)) if len(finite) else 1.0


def velocity(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / fps)

