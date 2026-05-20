import inspect
import unittest
from unittest.mock import patch

import main
import main_stage1_manual
import main_stage2_focus_assist
import main_stage3_autofocus


class UnifiedEntryTest(unittest.TestCase):
    def test_main_imports_unified_gui_app(self):
        source = inspect.getsource(main)

        self.assertIn("from gui_app import ProbeStationApp", source)
        self.assertNotIn("from app_gui import ProbeStationApp", source)

    def test_legacy_stage_entry_points_delegate_to_main(self):
        for module in (main_stage1_manual, main_stage2_focus_assist, main_stage3_autofocus):
            with self.subTest(module=module.__name__), patch.object(main, "main") as run_main:
                module.main()
                run_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
