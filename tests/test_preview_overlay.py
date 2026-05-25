import unittest

import numpy as np

from gui_app import CROSSHAIR_COLOR_RGB, draw_preview_crosshair


class PreviewCrosshairTest(unittest.TestCase):
    def test_crosshair_draws_thin_fluorescent_green_center_lines_on_a_copy(self):
        frame = np.zeros((9, 11, 3), dtype=np.uint8)

        overlaid = draw_preview_crosshair(frame)

        expected_color = np.asarray(CROSSHAIR_COLOR_RGB, dtype=np.uint8)
        np.testing.assert_array_equal(overlaid[4, 0], expected_color)
        np.testing.assert_array_equal(overlaid[0, 5], expected_color)
        np.testing.assert_array_equal(overlaid[4, 5], expected_color)
        np.testing.assert_array_equal(overlaid[3, 4], np.zeros(3, dtype=np.uint8))
        np.testing.assert_array_equal(frame, np.zeros_like(frame))


if __name__ == "__main__":
    unittest.main()
