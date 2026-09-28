"""MediaPipe Pose wrapper that keeps the rest of the app model-agnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
    "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]


@dataclass
class PoseFrame:
    landmarks: Optional[np.ndarray]
    # Array shape: (33, 4), columns x, y, z, visibility.


class MediaPipePoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "未安装 MediaPipe。请先运行: pip install -r requirements.txt"
            ) from exc
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "当前 MediaPipe 包不包含 solutions API；请安装 requirements.txt 中锁定的版本。"
            )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray) -> PoseFrame:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return PoseFrame(None)
        points = np.asarray(
            [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks.landmark],
            dtype=np.float64,
        )
        return PoseFrame(points)

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def landmarks_to_dict(points: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: points[i] for i, name in enumerate(LANDMARK_NAMES)}

