"""Local MediaPipe running-pose analysis CLI.

Usage:
    python main.py input/test.mp4 --output output
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from analysis.feedback import build_feedback
from analysis.metrics import compute_metrics
from analysis.stability import foot_cycle_analysis
from biomechanics.angles import angle_series
from biomechanics.foot_tracking import body_scale
from biomechanics.gait_events import FootEvents, detect_foot_events
from config import AnalysisConfig
from pose.mediapipe_pose import LANDMARK_NAMES, MediaPipePoseEstimator
from pose.smoothing import preprocess_landmarks
from visualization.plots import create_report
from visualization.video_overlay import draw_panel, draw_pose


IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="侧面跑步视频姿态与步态分析")
    parser.add_argument("video", type=Path, help="输入 .mp4/.mov/.avi 视频")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出目录")
    return parser.parse_args()


def validate_video(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"找不到输入视频: {path}")
    if path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
        raise ValueError("仅支持 .mp4、.mov、.avi 视频")


def extract_pose(video: Path) -> Tuple[np.ndarray, float, int, int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    with MediaPipePoseEstimator() as estimator:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            result = estimator.process(frame)
            frames.append(result.landmarks if result.landmarks is not None
                          else np.full((33, 4), np.nan))
            timestamps.append(timestamp)
            index += 1
            if index % max(1, int(fps * 2)) == 0:
                print(f"\r姿态检测: {index}/{total or '?'} 帧", end="", flush=True)
    capture.release()
    print()
    if not frames:
        raise RuntimeError("视频不包含可读取的帧")
    frame_times = np.asarray(timestamps, dtype=np.float64)
    if (not np.all(np.isfinite(frame_times)) or len(frame_times) < 2
            or np.count_nonzero(np.diff(frame_times) > 0) < len(frame_times) * 0.8):
        frame_times = np.arange(len(frames), dtype=np.float64) / fps
    else:
        frame_times -= frame_times[0]
    return np.stack(frames), fps, width, height, frame_times


def side_angles(points: np.ndarray, side: str) -> Dict[str, np.ndarray]:
    p = lambda name: points[:, IDX[f"{side}_{name}"], :]
    return {
        "knee": angle_series(p("hip"), p("knee"), p("ankle")),
        "hip": angle_series(p("shoulder"), p("hip"), p("knee")),
        "ankle": angle_series(p("knee"), p("ankle"), p("foot_index")),
    }


def save_landmarks(path: Path, raw: np.ndarray, smooth: np.ndarray, fps: float) -> None:
    fields = ["frame_id", "timestamp", "landmark", "x", "y", "z", "visibility",
              "x_smooth", "y_smooth", "z_smooth"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_id in range(len(raw)):
            for i, name in enumerate(LANDMARK_NAMES):
                writer.writerow({
                    "frame_id": frame_id,
                    "timestamp": round(frame_id / fps, 6),
                    "landmark": name,
                    "x": raw[frame_id, i, 0], "y": raw[frame_id, i, 1],
                    "z": raw[frame_id, i, 2], "visibility": raw[frame_id, i, 3],
                    "x_smooth": smooth[frame_id, i, 0],
                    "y_smooth": smooth[frame_id, i, 1],
                    "z_smooth": smooth[frame_id, i, 2],
                })


def save_timeline(path: Path, points: np.ndarray, fps: float, timestamps: np.ndarray,
                  angles: Dict[str, Dict[str, np.ndarray]],
                  event_map: Dict[int, str], foot_motion: Dict[str, object]) -> None:
    """Save compact frame data used by the synchronized browser player."""
    def series(values: np.ndarray) -> List[object]:
        return [round(float(value), 2) if np.isfinite(value) else None for value in values]

    frames = []
    for frame in points:
        frames.append([
            [round(float(value), 5) if np.isfinite(value) else None for value in point]
            for point in frame
        ])
    payload = {
        "fps": round(float(fps), 4),
        "frame_count": len(points),
        "timestamps": [round(float(value), 6) for value in timestamps],
        "landmarks": frames,
        "angles": {
            "left_knee": series(angles["left"]["knee"]),
            "right_knee": series(angles["right"]["knee"]),
            "left_hip": series(angles["left"]["hip"]),
            "right_hip": series(angles["right"]["hip"]),
        },
        "events": {str(frame): label for frame, label in event_map.items()},
        "foot_motion": {
            side: {"mean_path": foot_motion[side]["mean_path"],
                   "cycles": foot_motion[side]["cycles"]}
            for side in ("left", "right")
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def save_report_html(path: Path, result: Dict[str, object]) -> None:
    """Write a portable, expandable one-page summary next to the static chart."""
    report = result["report"]
    observations = "\n".join(
        f"<li>{html.escape(str(item))}</li>" for item in report["observations"]
    )
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(report['title']))}</title><style>
body{{margin:0;background:#0b100e;color:#edf3ee;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:840px;margin:5vh auto;padding:28px}}small{{color:#c9ff42;letter-spacing:.15em}}
h1{{font-size:clamp(36px,7vw,70px);line-height:1.1}}p,li{{color:#b7c3bb}}
details{{border:1px solid #334039;border-radius:12px;margin:25px 0;padding:20px}}
summary{{cursor:pointer;color:#c9ff42;font-weight:700}}li{{margin:12px 0}}
img{{display:block;max-width:100%;margin:20px auto;border-radius:8px}}
</style></head><body><main><small>PACE LAB · REPORT</small>
<h1>{html.escape(str(report['title']))}</h1><p>{html.escape(str(report['summary']))}</p>
<details open><summary>展开分析结论</summary><ul>{observations}</ul>
<p>{html.escape(str(report['method']))}</p></details>
<details><summary>查看图表报告</summary><img src="report.png" alt="分析图表"></details>
<p>{html.escape(str(result['disclaimer']))}</p></main></body></html>"""
    path.write_text(content, encoding="utf-8")


def save_annotated_video(source: Path, destination: Path, points: np.ndarray, fps: float,
                         width: int, height: int, angles: Dict[str, Dict[str, np.ndarray]],
                         event_map: Dict[int, str], metrics: Dict[str, object],
                         config: AnalysisConfig) -> None:
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {destination}")
    recent_event = ""
    recent_until = -1
    for i in range(len(points)):
        ok, frame = capture.read()
        if not ok:
            break
        if i in event_map:
            recent_event, recent_until = event_map[i], i + int(config.event_display_seconds * fps)
        event_text = recent_event if i <= recent_until else ""
        draw_pose(frame, points[i], config.min_visibility)
        values = [angles[s]["knee"][i] for s in ("left", "right")]
        finite = [v for v in values if np.isfinite(v)]
        draw_panel(frame, float(np.mean(finite)) if finite else None,
                   metrics.get("cadence_steps_per_min"), event_text, (i + 1) / len(points))
        writer.write(frame)
    capture.release()
    writer.release()


def analyze(video: Path, output: Path, config: AnalysisConfig) -> Dict[str, object]:
    validate_video(video)
    output.mkdir(parents=True, exist_ok=True)
    raw, fps, width, height, timestamps = extract_pose(video)
    detected_ratio = float(np.isfinite(raw[:, :, 0]).any(axis=1).mean())
    if detected_ratio < 0.2:
        raise RuntimeError(f"人体检测有效帧仅 {detected_ratio:.1%}，请使用无遮挡的固定侧面全身视频")
    smooth = preprocess_landmarks(raw, fps, config.min_visibility,
                                  config.smoothing_window_seconds)
    angles = {side: side_angles(smooth, side) for side in ("left", "right")}
    events: Dict[str, FootEvents] = {}
    for side in ("left", "right"):
        foot = smooth[:, IDX[f"{side}_foot_index"], :]
        events[side] = detect_foot_events(
            foot[:, 0], foot[:, 1], fps, config.min_event_interval_seconds,
            config.ground_percentile, config.strike_velocity_tolerance,
        )
    scale = np.mean([
        body_scale(smooth[:, IDX[f"{s}_shoulder"]], smooth[:, IDX[f"{s}_hip"]],
                   smooth[:, IDX[f"{s}_knee"]], smooth[:, IDX[f"{s}_ankle"]])
        for s in ("left", "right")
    ])
    metrics = compute_metrics(
        fps, len(smooth), events["left"].strikes, events["right"].strikes,
        angles["left"]["knee"], angles["right"]["knee"],
        angles["left"]["hip"], angles["right"]["hip"],
        smooth[:, IDX["left_foot_index"]], smooth[:, IDX["right_foot_index"]],
        smooth[:, IDX["left_hip"]], smooth[:, IDX["right_hip"]], scale,
        config.min_stride_seconds, config.max_stride_seconds,
    )
    metrics.update({
        "source_video": str(video.resolve()), "fps": round(fps, 3),
        "frame_count": len(smooth), "pose_detection_rate": round(detected_ratio, 3),
        "left_foot_strikes": events["left"].strikes,
        "right_foot_strikes": events["right"].strikes,
        "left_toe_offs": events["left"].toe_offs,
        "right_toe_offs": events["right"].toe_offs,
    })
    feedback = build_feedback(metrics, config)
    foot_motion = foot_cycle_analysis(
        smooth, {side: events[side].strikes for side in ("left", "right")},
        scale, config.min_visibility,
    )
    metrics["foot_path_dispersion_body_ratio"] = foot_motion["overall_dispersion_body_ratio"]
    foot_summary = {
        **{side: {"cycle_count": foot_motion[side]["cycle_count"],
                  "dispersion_body_ratio": foot_motion[side]["dispersion_body_ratio"]}
           for side in ("left", "right")},
        "overall_dispersion_body_ratio": foot_motion["overall_dispersion_body_ratio"],
        "assessment": foot_motion["assessment"],
        "explanation": foot_motion["explanation"],
        "method": foot_motion["method"],
    }
    result = {"metrics": metrics, "feedback": feedback,
              "foot_motion": foot_summary,
              "report": {
                  "title": "跑姿分析简报",
                  "summary": foot_motion["explanation"],
                  "observations": [
                      f"步频：{metrics['cadence_steps_per_min'] if metrics['cadence_steps_per_min'] is not None else '数据不足'} 步/分钟。",
                      f"足部轨迹重复性：{foot_motion['assessment']}；左右有效周期分别为 {foot_motion['left']['cycle_count']} 和 {foot_motion['right']['cycle_count']} 个。",
                      *feedback,
                  ],
                  "method": foot_motion["method"],
              },
              "disclaimer": "二维视频估算结果，仅供运动观察，不用于医疗诊断。"}
    save_landmarks(output / "landmarks.csv", raw, smooth, fps)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    save_report_html(output / "report.html", result)
    event_frames: Dict[int, str] = {}
    for side in ("left", "right"):
        for i in events[side].strikes:
            event_frames[i] = f"{side.title()} foot strike"
        for i in events[side].toe_offs:
            event_frames[i] = f"{side.title()} toe-off"
    save_timeline(output / "timeline.json", smooth, fps, timestamps, angles, event_frames, foot_motion)
    save_annotated_video(video, output / "annotated.mp4", smooth, fps, width, height,
                         angles, event_frames, metrics, config)
    times = np.arange(len(smooth)) / fps
    create_report(
        output / "report.png", times,
        {"left_knee": angles["left"]["knee"], "right_knee": angles["right"]["knee"]},
        {"left": smooth[:, IDX["left_foot_index"], 1],
         "right": smooth[:, IDX["right_foot_index"], 1]},
        {"left_strikes": events["left"].strikes,
         "right_strikes": events["right"].strikes}, metrics, feedback,
    )
    return result


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args.video, args.output, AnalysisConfig())
    except (ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n完成，结果位于: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
