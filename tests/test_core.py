import unittest

import numpy as np

from biomechanics.angles import angle_series
from biomechanics.gait_events import detect_foot_events
from analysis.metrics import compute_metrics
from analysis.stability import foot_cycle_analysis


class CoreTests(unittest.TestCase):
    def test_angle(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(float(angle_series(a, b, c)[0]), 90.0)

    def test_periodic_strikes(self):
        fps = 30.0
        t = np.arange(300) / fps
        y = 0.7 + 0.12 * np.cos(2 * np.pi * 1.5 * t)
        x = 0.5 + 0.08 * np.sin(2 * np.pi * 1.5 * t)
        events = detect_foot_events(x, y, fps)
        self.assertGreaterEqual(len(events.strikes), 10)
        intervals = np.diff(events.strikes) / fps
        self.assertAlmostEqual(float(np.median(intervals)), 1 / 1.5, delta=0.08)

    def test_missing_angles_are_json_safe(self):
        empty = np.full(30, np.nan)
        xy = np.full((30, 4), np.nan)
        result = compute_metrics(
            30.0, 30, [], [], empty, empty, empty, empty, xy, xy, xy, xy,
            1.0, 0.35, 2.0,
        )
        self.assertIsNone(result["knee_rom_degrees"])
        self.assertIsNone(result["hip_rom_degrees"])

    def test_foot_cycle_repeatability_and_insufficient_data(self):
        frames = 121
        points = np.full((frames, 33, 4), np.nan)
        phase = np.arange(frames) % 40 / 40
        for side, foot_index, hip_index in (("left", 31, 23), ("right", 32, 24)):
            points[:, hip_index, :2] = [0.5, 0.5]
            points[:, hip_index, 3] = 1
            points[:, foot_index, 0] = 0.5 + 0.15 * np.sin(2 * np.pi * phase)
            points[:, foot_index, 1] = 0.7 + 0.10 * np.cos(2 * np.pi * phase)
            points[:, foot_index, 3] = 1
        result = foot_cycle_analysis(points, {"left": [0, 40, 80, 120], "right": [0, 40, 80, 120]}, 0.2)
        self.assertEqual(result["left"]["cycle_count"], 3)
        self.assertLess(result["overall_dispersion_body_ratio"], 0.01)
        insufficient = foot_cycle_analysis(points, {"left": [0, 40], "right": []}, 0.2)
        self.assertEqual(insufficient["assessment"], "数据不足")


if __name__ == "__main__":
    unittest.main()
