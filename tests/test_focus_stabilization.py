import unittest

import numpy as np

import focus_metrics
from focus_metrics import stabilize_frame_translation


class FocusStabilizationTest(unittest.TestCase):
    @unittest.skipIf(focus_metrics.cv2 is None, "OpenCV is not installed")
    def test_translation_stabilizer_reduces_shifted_frame_difference(self):
        reference = np.zeros((96, 96), dtype=np.uint8)
        reference[30:60, 36:66] = 220
        shifted = np.roll(np.roll(reference, 4, axis=0), -5, axis=1)

        stabilized, info = stabilize_frame_translation(reference, shifted)

        raw_error = float(np.mean(np.abs(reference.astype(float) - shifted.astype(float))))
        stabilized_error = float(np.mean(np.abs(reference.astype(float) - stabilized.astype(float))))
        self.assertLess(stabilized_error, raw_error)
        self.assertGreater(info["response"], 0.1)


if __name__ == "__main__":
    unittest.main()
