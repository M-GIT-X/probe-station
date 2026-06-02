import unittest

from sample_plane import SamplePlanePoint, fit_sample_plane
from scan_plan import ScanBounds, point_in_polygon
from stitching_calibration import (
    StitchingCalibration,
    build_in_bounds_trial_plan,
    calibration_from_shifts,
    estimate_calibration_from_frames,
)


class StitchingCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.bounds = ScanBounds(min_x=0, max_x=1000, min_y=100, max_y=900)
        self.plane = fit_sample_plane(
            [
                SamplePlanePoint("c1", 0, 100, 100),
                SamplePlanePoint("c2", 1000, 100, 110),
                SamplePlanePoint("c3", 1000, 900, 130),
                SamplePlanePoint("c4", 0, 900, 120),
            ]
        )

    def test_trial_plan_keeps_reference_x_and_y_moves_inside_bounds(self):
        trial = build_in_bounds_trial_plan(self.bounds, self.plane)

        for point in (trial.reference, trial.x_trial, trial.y_trial):
            self.assertGreaterEqual(point.x, self.bounds.min_x)
            self.assertLessEqual(point.x, self.bounds.max_x)
            self.assertGreaterEqual(point.y, self.bounds.min_y)
            self.assertLessEqual(point.y, self.bounds.max_y)
            self.assertEqual(point.z, self.plane.z_at(point.x, point.y))
        self.assertGreater(trial.x_step_pulses, 0)
        self.assertGreater(trial.y_step_pulses, 0)

    def test_calibration_turns_observed_content_shift_into_signed_tile_placement_scale(self):
        calibration = calibration_from_shifts(
            x_step_pulses=100,
            y_step_pulses=80,
            x_frame_shift=(-32.0, 0.4),
            y_frame_shift=(0.3, -20.0),
            frame_width=640,
            frame_height=480,
            overlap_percent=25.0,
            confidence=0.95,
        )

        self.assertAlmostEqual(calibration.x_pixels_per_pulse, 0.32)
        self.assertAlmostEqual(calibration.y_pixels_per_pulse, 0.25)
        self.assertEqual(calibration.overlap_percent, 25.0)
        self.assertIsInstance(calibration, StitchingCalibration)

    def test_calibration_rejects_shift_that_is_too_small_to_measure(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            calibration_from_shifts(
                x_step_pulses=100,
                y_step_pulses=80,
                x_frame_shift=(-0.2, 0.0),
                y_frame_shift=(0.0, -20.0),
                frame_width=640,
                frame_height=480,
                overlap_percent=25.0,
                confidence=0.9,
            )

    def test_calibration_rejects_cross_axis_dominated_motion(self):
        with self.assertRaisesRegex(ValueError, "axis"):
            calibration_from_shifts(
                x_step_pulses=100,
                y_step_pulses=80,
                x_frame_shift=(-8.0, 18.0),
                y_frame_shift=(0.5, -20.0),
                frame_width=640,
                frame_height=480,
                overlap_percent=25.0,
                confidence=0.9,
            )

    def test_trial_points_remain_inside_recorded_quadrilateral_not_just_bounding_box(self):
        boundary = [(500, 100), (1000, 500), (500, 900), (0, 500)]

        trial = build_in_bounds_trial_plan(self.bounds, self.plane, boundary_points=boundary)

        for point in (trial.reference, trial.x_trial, trial.y_trial):
            self.assertTrue(point_in_polygon(point.x, point.y, boundary))

    def test_phase_correlation_estimates_scale_from_in_memory_trial_frames(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV is not installed")
        rng = np.random.default_rng(9)
        reference = rng.integers(0, 255, size=(120, 160), dtype=np.uint8)
        x_trial = cv2.warpAffine(reference, np.float32([[1, 0, -12], [0, 1, 0]]), (160, 120))
        y_trial = cv2.warpAffine(reference, np.float32([[1, 0, 0], [0, 1, -8]]), (160, 120))
        trial = build_in_bounds_trial_plan(self.bounds, self.plane)

        calibration = estimate_calibration_from_frames(reference, x_trial, y_trial, trial, 25.0)

        self.assertAlmostEqual(calibration.x_pixels_per_pulse, 12 / trial.x_step_pulses, delta=0.01)
        self.assertAlmostEqual(calibration.y_pixels_per_pulse, 8 / trial.y_step_pulses, delta=0.01)


if __name__ == "__main__":
    unittest.main()
