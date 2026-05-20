import unittest

from gui_app import APP_TITLE, Mode


class AppModeTitleTest(unittest.TestCase):
    def test_app_title_is_unified(self):
        self.assertEqual(APP_TITLE, "Three-Axis Probe Station")

    def test_modes_are_three_explicit_gui_modes(self):
        self.assertEqual([mode.value for mode in Mode], ["Manual Mode", "Manual Focus Assist", "Auto Focus"])


if __name__ == "__main__":
    unittest.main()
