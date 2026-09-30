"""Pose backends with a common normalized MediaPipe-compatible layout."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import urllib.request

import cv2
import numpy as np

from pose.mediapipe_pose import MediaPipePoseEstimator, PoseFrame

MODEL_NAMES = {"mediapipe": "MediaPipe", "rtmpose": "RTMPose", "movenet": "MoveNet"}
# COCO 17 -> MediaPipe 33. Missing toes, heels, hands and face detail stay missing.
COCO_TO_MEDIAPIPE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MOVENET_URL = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
_RTMPOSE_LOAD_LOCK = threading.Lock()


def validate_model(model: str) -> str:
    if not isinstance(model, str) or model not in MODEL_NAMES:
        raise ValueError("不支持的姿态模型，请选择 MediaPipe、RTMPose 或 MoveNet")
    return model


def coco_landmarks(xy: np.ndarray, scores: np.ndarray) -> PoseFrame:
    points = np.full((33, 4), np.nan, dtype=np.float64)
    points[:, 3] = 0
    points[COCO_TO_MEDIAPIPE, :2] = xy
    # COCO models are 2D; z=0 is only a placeholder, never a depth measurement.
    points[COCO_TO_MEDIAPIPE, 2] = 0
    points[COCO_TO_MEDIAPIPE, 3] = np.clip(scores, 0, 1)
    if np.count_nonzero(scores[5:] >= 0.3) < 4:
        return PoseFrame(None)
    return PoseFrame(points)


class CocoEstimator:
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class RTMPoseEstimator(CocoEstimator):
    def __init__(self) -> None:
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise RuntimeError("RTMPose 依赖缺失，请运行 pip install -r requirements.txt") from exc
        try:
            with _RTMPOSE_LOAD_LOCK:
                self._body = Body(mode="lightweight", to_openpose=False,
                                  backend="onnxruntime", device="cpu")
        except Exception as exc:
            raise RuntimeError(f"RTMPose 加载失败（首次使用需要联网下载模型）：{exc}") from exc

    def process(self, frame: np.ndarray) -> PoseFrame:
        keypoints, scores = self._body(frame)
        if len(keypoints) == 0:
            return PoseFrame(None)
        # This demo analyzes one runner: select the most confident body.
        index = int(np.argmax(np.mean(scores[:, 5:], axis=1)))
        height, width = frame.shape[:2]
        return coco_landmarks(keypoints[index] / [width, height], scores[index])

    def close(self) -> None:
        self._body = None


def movenet_path() -> Path:
    override = os.environ.get("PACE_MOVENET_MODEL")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise RuntimeError(f"找不到 MoveNet 模型：{path}")
        return path
    cache = Path(os.environ.get("PACE_DATA_DIR", str(Path(__file__).resolve().parents[1] / "output"))) / "models"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "movenet-lightning-float16.tflite"
    if path.is_file():
        return path
    temporary = None
    try:
        with urllib.request.urlopen(MOVENET_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
                temporary = Path(handle.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        with temporary.open("rb") as handle:
            if handle.read(8)[4:8] != b"TFL3":
                raise RuntimeError("下载内容不是 TFLite 模型")
        temporary.replace(path)
    except Exception as exc:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise RuntimeError("MoveNet 下载失败，请检查网络，或设置 PACE_MOVENET_MODEL 指向本地 .tflite 文件") from exc
    return path


class MoveNetEstimator(CocoEstimator):
    def __init__(self) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError as exc:
                raise RuntimeError("MoveNet 依赖缺失，请安装 ai-edge-litert（Intel Mac 可安装 tensorflow）") from exc
        try:
            self._interpreter = Interpreter(model_path=str(movenet_path()), num_threads=2)
            self._interpreter.allocate_tensors()
            self._input = self._interpreter.get_input_details()[0]
            self._output = self._interpreter.get_output_details()[0]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"MoveNet 模型加载失败：{exc}") from exc

    def process(self, frame: np.ndarray) -> PoseFrame:
        height, width = frame.shape[:2]
        _, target_h, target_w, _ = self._input["shape"]
        scale = min(target_w / width, target_h / height)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (target_w - resized_w) // 2, (target_h - resized_h) // 2
        image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        image[top:top + resized_h, left:left + resized_w] = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (resized_w, resized_h))
        self._interpreter.set_tensor(self._input["index"], image[None].astype(self._input["dtype"]))
        self._interpreter.invoke()
        keypoints = self._interpreter.get_tensor(self._output["index"]).reshape(17, 3)
        xy = np.column_stack(((keypoints[:, 1] * target_w - left) / resized_w,
                              (keypoints[:, 0] * target_h - top) / resized_h))
        scores = keypoints[:, 2].copy()
        scores[np.any((xy < 0) | (xy > 1), axis=1)] = 0
        return coco_landmarks(xy, scores)

    def close(self) -> None:
        self._interpreter = None


def create_estimator(model: str = "mediapipe"):
    validate_model(model)
    if model == "rtmpose":
        return RTMPoseEstimator()
    if model == "movenet":
        return MoveNetEstimator()
    return MediaPipePoseEstimator()
