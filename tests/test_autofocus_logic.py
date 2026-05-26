import unittest

from gui_app import (
    AutofocusMode,
    AutofocusParams,
    AutofocusPassPlan,
    AutofocusSamplePoint,
    SAFE_MAX_AUTOFOCUS_RANGE,
    SAFE_MAX_AUTOFOCUS_SPEED,
    SAFE_MAX_AUTOFOCUS_STEP,
    SAFE_MAX_MANUAL_SPEED,
    SAFE_MAX_MANUAL_STEP,
    build_autofocus_pass_plan,
    build_scan_offsets,
    clamp_autofocus_params,
    clamp_manual_motion_params,
    logical_direction_to_controller_direction,
    manual_shortcut_mapping,
    select_final_autofocus_point,
)


class DirectionMappingTest(unittest.TestCase):
    def test_x_direction_is_swapped_after_real_machine_feedback(self):
        self.assertEqual(logical_direction_to_controller_direction("X", +1), +1)
        self.assertEqual(logical_direction_to_controller_direction("X", -1), -1)
        self.assertEqual(logical_direction_to_controller_direction("Y", +1), -1)
        self.assertEqual(logical_direction_to_controller_direction("Y", -1), +1)
        self.assertEqual(logical_direction_to_controller_direction("Z", +1), +1)
        self.assertEqual(logical_direction_to_controller_direction("Z", -1), -1)

    def test_manual_xy_shortcuts_follow_operator_direction_feedback(self):
        self.assertEqual(manual_shortcut_mapping("a"), ("X", +1))
        self.assertEqual(manual_shortcut_mapping("d"), ("X", -1))
        self.assertEqual(manual_shortcut_mapping("w"), ("Y", -1))
        self.assertEqual(manual_shortcut_mapping("s"), ("Y", +1))


class AutofocusLogicTest(unittest.TestCase):
    def test_safe_mode_allows_manual_motion_up_to_2000_pulses_and_full_protocol_speed(self):
        pulses, speed, changed = clamp_manual_motion_params(5000, 200)

        self.assertTrue(changed)
        self.assertEqual(pulses, SAFE_MAX_MANUAL_STEP)
        self.assertEqual(speed, SAFE_MAX_MANUAL_SPEED)
        self.assertEqual(SAFE_MAX_MANUAL_STEP, 2000)
        self.assertEqual(SAFE_MAX_MANUAL_SPEED, 100)

    def test_safe_mode_allows_autofocus_range_step_up_to_2000_and_full_protocol_speed(self):
        params = AutofocusParams(
            scan_range=5000,
            scan_step=5000,
            autofocus_speed=200,
            settle_seconds=0.5,
            sample_seconds=1.5,
            near_best_ratio=0.96,
        )

        clamped, changed = clamp_autofocus_params(params)

        self.assertTrue(changed)
        self.assertEqual(clamped.scan_range, SAFE_MAX_AUTOFOCUS_RANGE)
        self.assertEqual(clamped.scan_step, SAFE_MAX_AUTOFOCUS_STEP)
        self.assertEqual(clamped.autofocus_speed, SAFE_MAX_AUTOFOCUS_SPEED)
        self.assertEqual(SAFE_MAX_AUTOFOCUS_RANGE, 2000)
        self.assertEqual(SAFE_MAX_AUTOFOCUS_STEP, 2000)
        self.assertEqual(SAFE_MAX_AUTOFOCUS_SPEED, 100)

    def test_scan_range_is_half_range_around_current_position(self):
        self.assertEqual(build_scan_offsets(20, 10), [-20, -10, 0, 10, 20])

    def test_scan_offsets_include_positive_range_even_when_step_does_not_land_on_it(self):
        self.assertEqual(build_scan_offsets(20, 6), [-20, -14, -8, -2, 4, 10, 16, 20])

    def test_semi_auto_uses_user_range_and_step_as_single_pass(self):
        params = AutofocusParams(scan_range=40, scan_step=8)

        self.assertEqual(
            build_autofocus_pass_plan(params, AutofocusMode.SEMI),
            [AutofocusPassPlan(name="semi", scan_range=40, scan_step=8)],
        )

    def test_full_auto_builds_coarse_to_fine_passes_from_range_only(self):
        params = AutofocusParams(scan_range=80, scan_step=99)

        passes = build_autofocus_pass_plan(params, AutofocusMode.FULL)

        self.assertGreater(len(passes), 1)
        self.assertEqual(passes[0].scan_range, 80)
        self.assertLess(passes[-1].scan_range, passes[0].scan_range)
        self.assertEqual(passes[1].scan_step, max(1, passes[0].scan_step // 2))
        self.assertEqual(passes[2].scan_step, max(1, passes[1].scan_step // 2))

    def test_final_point_prefers_stability_inside_near_best_band(self):
        points = [
            AutofocusSamplePoint(offset=-10, score=90.0, iqr=1.0, frame_count=10),
            AutofocusSamplePoint(offset=0, score=96.0, iqr=4.0, frame_count=10),
            AutofocusSamplePoint(offset=5, score=100.0, iqr=30.0, frame_count=10),
        ]

        selected = select_final_autofocus_point(points, near_best_ratio=0.96)

        self.assertEqual(selected.offset, 0)

    def test_final_point_uses_smaller_abs_offset_when_stability_is_close(self):
        points = [
            AutofocusSamplePoint(offset=-10, score=98.0, iqr=2.0, frame_count=10),
            AutofocusSamplePoint(offset=3, score=100.0, iqr=2.02, frame_count=10),
        ]

        selected = select_final_autofocus_point(points, near_best_ratio=0.96)

        self.assertEqual(selected.offset, 3)


if __name__ == "__main__":
    unittest.main()
