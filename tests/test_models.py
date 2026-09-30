import unittest

import numpy as np

from pose.backends import COCO_TO_MEDIAPIPE, MoveNetEstimator, coco_landmarks, validate_model


class ModelMappingTests(unittest.TestCase):
    def test_mapping_preserves_joints_and_does_not_invent_feet(self):
        xy = np.arange(34).reshape(17, 2) / 40
        points = coco_landmarks(xy, np.ones(17)).landmarks
        np.testing.assert_allclose(points[COCO_TO_MEDIAPIPE, :2], xy)
        self.assertTrue(np.isnan(points[[29, 30, 31, 32], :3]).all())
        self.assertTrue((points[[29, 30, 31, 32], 3] == 0).all())

    def test_low_confidence_is_not_detected(self):
        self.assertIsNone(coco_landmarks(np.zeros((17, 2)), np.zeros(17)).landmarks)

    def test_movenet_removes_letterbox_padding(self):
        estimator = MoveNetEstimator.__new__(MoveNetEstimator)
        estimator._input = {"shape": [1, 192, 192, 3], "index": 0, "dtype": np.uint8}
        estimator._output = {"index": 1}
        class Interpreter:
            def set_tensor(self, index, image):
                self.image = image
            def invoke(self):
                pass
            def get_tensor(self, index):
                return np.tile([.375, .25, .9], (1, 1, 17, 1))
        estimator._interpreter = Interpreter()
        points = estimator.process(np.zeros((100, 200, 3), dtype=np.uint8)).landmarks
        np.testing.assert_allclose(points[23, :2], [.25, .25])
        self.assertEqual(estimator._interpreter.image.shape, (1, 192, 192, 3))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_model('unknown')


