import unittest

import numpy as np

from gui_app import CROSSHAIR_COLOR_RGB, draw_preview_crosshair, stage_delta_from_preview_delta
from stitching_calibration import calibration_from_shifts


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

    def test_mouse_mode_draws_small_crosshair_at_cursor_without_full_frame_lines(self):
        frame = np.zeros((15, 15, 3), dtype=np.uint8)

        overlaid = draw_preview_crosshair(frame, cursor_position=(4, 10), include_center=False, cursor_radius=2)

        expected_color = np.asarray(CROSSHAIR_COLOR_RGB, dtype=np.uint8)
        np.testing.assert_array_equal(overlaid[10, 4], expected_color)
        np.testing.assert_array_equal(overlaid[10, 2], expected_color)
        np.testing.assert_array_equal(overlaid[8, 4], expected_color)
        np.testing.assert_array_equal(overlaid[10, 0], np.zeros(3, dtype=np.uint8))
        np.testing.assert_array_equal(overlaid[7, 4], np.zeros(3, dtype=np.uint8))

    def test_stage_delta_from_preview_delta_uses_signed_axis_calibration(self):
        calibration = calibration_from_shifts(
            x_step_pulses=10,
            y_step_pulses=10,
            x_frame_shift=(-5.0, 0.0),
            y_frame_shift=(0.0, -2.5),
            frame_width=100,
            frame_height=80,
            overlap_percent=0.0,
            confidence=0.9,
        )

        self.assertEqual(stage_delta_from_preview_delta(25, -10, calibration), (50, -40))


if __name__ == "__main__":
    unittest.main()
