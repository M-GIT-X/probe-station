import unittest

from gui_app import mission_log_message_for_event, should_ignore_axis_shortcut, should_return_focus_to_root_on_enter


class FakeWidget:
    def __init__(self, widget_class):
        self._widget_class = widget_class

    def winfo_class(self):
        return self._widget_class


class KeyboardShortcutTest(unittest.TestCase):
    def test_axis_shortcuts_are_ignored_while_typing_in_text_inputs(self):
        for widget_class in ("Entry", "TEntry", "Combobox", "TCombobox", "Spinbox", "TSpinbox"):
            with self.subTest(widget_class=widget_class):
                self.assertTrue(should_ignore_axis_shortcut(FakeWidget(widget_class)))

    def test_axis_shortcuts_work_outside_text_inputs(self):
        self.assertFalse(should_ignore_axis_shortcut(FakeWidget("Button")))
        self.assertFalse(should_ignore_axis_shortcut(None))

    def test_enter_returns_focus_to_manual_control_from_text_inputs(self):
        for widget_class in ("Entry", "TEntry", "Combobox", "TCombobox", "Spinbox", "TSpinbox"):
            with self.subTest(widget_class=widget_class):
                self.assertTrue(should_return_focus_to_root_on_enter(FakeWidget(widget_class)))

    def test_enter_does_not_steal_focus_from_regular_controls(self):
        self.assertFalse(should_return_focus_to_root_on_enter(FakeWidget("Button")))
        self.assertFalse(should_return_focus_to_root_on_enter(None))

    def test_mission_log_includes_only_key_events(self):
        self.assertEqual(
            mission_log_message_for_event("camera_opened", "index 0 backend DSHOW"),
            "CAMERA ONLINE. Optical feed established: index 0 backend DSHOW.",
        )
        self.assertIsNone(mission_log_message_for_event("positions", {"X": 1}))

    def test_mission_log_announces_stitching_scan_and_mosaic_lifecycle(self):
        self.assertEqual(
            mission_log_message_for_event("stitch_scan_completed"),
            "SCAN ACQUISITION COMPLETE. All image tiles secured.",
        )
        self.assertEqual(
            mission_log_message_for_event("stitch_assembling"),
            "MOSAIC ASSEMBLY IN PROGRESS. Processing captured tiles.",
        )
        self.assertEqual(
            mission_log_message_for_event("stitch_completed", "result.png"),
            "MOSAIC COMPLETE. Composite image saved: result.png.",
        )


if __name__ == "__main__":
    unittest.main()
