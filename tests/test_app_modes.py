import unittest

from gui_app import (
    APP_TITLE,
    AUTOFOCUS_CURVE_WIDTH,
    DEFAULT_WINDOW_GEOMETRY,
    EMERGENCY_STOP_BACKGROUND_COLOR,
    EMERGENCY_STOP_FOREGROUND_COLOR,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    MISSION_LOG_BACKGROUND_COLOR,
    MISSION_LOG_FOREGROUND_COLOR,
    Mode,
    PREVIEW_PLACEHOLDER_TEXT,
    STITCHING_PLOT_SIZE,
    VIDEO_PREVIEW_MAX_HEIGHT,
    VIDEO_PREVIEW_MAX_WIDTH,
    mode_panel_spec,
    stitching_geometry_input_fields,
)


class AppModeTitleTest(unittest.TestCase):
    def test_app_title_is_unified(self):
        self.assertEqual(APP_TITLE, "Three-Axis Probe Station")

    def test_modes_are_explicit_gui_modes(self):
        self.assertEqual(
            [mode.value for mode in Mode],
            ["Manual Mode", "Auto Focus", "Image Stitching"],
        )

    def test_mode_specs_show_different_function_sections(self):
        manual = mode_panel_spec(Mode.MANUAL)
        autofocus = mode_panel_spec(Mode.AUTO_FOCUS)
        stitching = mode_panel_spec(Mode.IMAGE_STITCHING)

        self.assertEqual(manual.visible_sections, ("manual",))
        self.assertEqual(autofocus.visible_sections, ("autofocus",))
        self.assertEqual(stitching.visible_sections, ("image_stitching",))
        self.assertIn("Start Autofocus", autofocus.primary_actions)
        self.assertIn("Record Corner", stitching.primary_actions)
        self.assertNotIn("Run Offline Stitch", stitching.primary_actions)
        self.assertIn("Stitched mosaic", stitching.status_fields)

    def test_window_and_video_defaults_support_a_larger_primary_preview(self):
        self.assertEqual(DEFAULT_WINDOW_GEOMETRY, "1580x1180")
        self.assertLessEqual(MIN_WINDOW_WIDTH, 960)
        self.assertLessEqual(MIN_WINDOW_HEIGHT, 620)
        self.assertGreaterEqual(VIDEO_PREVIEW_MAX_WIDTH, 680)
        self.assertGreaterEqual(VIDEO_PREVIEW_MAX_HEIGHT, 425)
        self.assertIn("CAMERA OFFLINE", PREVIEW_PLACEHOLDER_TEXT)

    def test_dynamic_view_dimensions_follow_camera_preview_width(self):
        self.assertEqual(STITCHING_PLOT_SIZE, VIDEO_PREVIEW_MAX_WIDTH)
        self.assertGreater(AUTOFOCUS_CURVE_WIDTH, VIDEO_PREVIEW_MAX_WIDTH)
        self.assertLessEqual(AUTOFOCUS_CURVE_WIDTH - VIDEO_PREVIEW_MAX_WIDTH, 24)

    def test_operational_console_and_emergency_button_colors_are_explicit(self):
        self.assertEqual((MISSION_LOG_BACKGROUND_COLOR, MISSION_LOG_FOREGROUND_COLOR), ("#000000", "#39ff14"))
        self.assertEqual((EMERGENCY_STOP_BACKGROUND_COLOR, EMERGENCY_STOP_FOREGROUND_COLOR), ("#d45d67", "#ffffff"))

    def test_stitching_geometry_only_asks_operator_for_overlap(self):
        self.assertEqual(stitching_geometry_input_fields(), ("Overlap %",))


if __name__ == "__main__":
    unittest.main()
