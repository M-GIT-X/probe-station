import unittest

from gui_app import should_ignore_axis_shortcut


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


if __name__ == "__main__":
    unittest.main()
