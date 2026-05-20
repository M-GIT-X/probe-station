import unittest

from gui_app import (
    APP_TITLE,
    DEFAULT_WINDOW_GEOMETRY,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    Mode,
    VIDEO_PREVIEW_MAX_HEIGHT,
    VIDEO_PREVIEW_MAX_WIDTH,
    mode_panel_spec,
)


class AppModeTitleTest(unittest.TestCase):
    def test_app_title_is_unified(self):
        self.assertEqual(APP_TITLE, "Three-Axis Probe Station")

    def test_modes_are_three_explicit_gui_modes(self):
        self.assertEqual([mode.value for mode in Mode], ["Manual Mode", "Manual Focus Assist", "Auto Focus"])

    def test_mode_specs_show_different_function_sections(self):
        manual = mode_panel_spec(Mode.MANUAL)
        assist = mode_panel_spec(Mode.FOCUS_ASSIST)
        autofocus = mode_panel_spec(Mode.AUTO_FOCUS)

        self.assertEqual(manual.visible_sections, ("manual",))
        self.assertEqual(assist.visible_sections, ("manual", "focus_assist"))
        self.assertEqual(autofocus.visible_sections, ("manual", "autofocus"))
        self.assertIn("Start Autofocus", autofocus.primary_actions)
        self.assertIn("Best Z", assist.status_fields)

    def test_window_and_video_defaults_fit_common_small_windows_screen(self):
        self.assertEqual(DEFAULT_WINDOW_GEOMETRY, "1120x700")
        self.assertLessEqual(MIN_WINDOW_WIDTH, 960)
        self.assertLessEqual(MIN_WINDOW_HEIGHT, 620)
        self.assertLessEqual(VIDEO_PREVIEW_MAX_WIDTH, 420)
        self.assertLessEqual(VIDEO_PREVIEW_MAX_HEIGHT, 240)


if __name__ == "__main__":
    unittest.main()
