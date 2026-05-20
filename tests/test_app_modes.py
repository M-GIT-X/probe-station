import unittest

from app_gui import app_stage_title


class AppModeTitleTest(unittest.TestCase):
    def test_stage_titles_are_distinct(self):
        self.assertEqual(app_stage_title(False, False), "Stage 1 Manual Control")
        self.assertEqual(app_stage_title(True, False), "Stage 2 Manual Focus Assist")
        self.assertEqual(app_stage_title(True, True), "Stage 3 Conservative Full Scan Autofocus")


if __name__ == "__main__":
    unittest.main()
