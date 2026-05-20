import unittest

from gui_app import ManualFocusAssistState


class ManualFocusAssistStateTest(unittest.TestCase):
    def test_first_valid_sample_becomes_best(self):
        state = ManualFocusAssistState(recording=True)

        updated = state.record_sample(12.5, z_abs=100, z_rel=-20, timestamp=1.0)

        self.assertTrue(updated)
        self.assertEqual(state.best_focus_index, 12.5)
        self.assertEqual(state.best_z_abs, 100)
        self.assertEqual(state.best_z_rel, -20)
        self.assertEqual(state.best_timestamp, 1.0)

    def test_small_jitter_does_not_replace_best(self):
        state = ManualFocusAssistState(recording=True)
        state.record_sample(100.0, z_abs=10, z_rel=10, timestamp=1.0)

        updated = state.record_sample(100.4, z_abs=11, z_rel=11, timestamp=2.0)

        self.assertFalse(updated)
        self.assertEqual(state.best_focus_index, 100.0)
        self.assertEqual(state.best_z_abs, 10)

    def test_more_than_threshold_updates_best(self):
        state = ManualFocusAssistState(recording=True)
        state.record_sample(100.0, z_abs=10, z_rel=10, timestamp=1.0)

        updated = state.record_sample(100.6, z_abs=11, z_rel=11, timestamp=2.0)

        self.assertTrue(updated)
        self.assertEqual(state.best_focus_index, 100.6)
        self.assertEqual(state.best_z_abs, 11)

    def test_reset_keeps_recording_optionally(self):
        state = ManualFocusAssistState(recording=True)
        state.record_sample(50.0, z_abs=1, z_rel=1, timestamp=1.0)

        state.reset_best(keep_recording=True)

        self.assertTrue(state.recording)
        self.assertIsNone(state.best_focus_index)
        self.assertIsNone(state.best_z_abs)


if __name__ == "__main__":
    unittest.main()
