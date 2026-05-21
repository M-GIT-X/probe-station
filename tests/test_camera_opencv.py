import unittest

from camera_opencv import choose_best_exposure_result, exposure_tuning_candidates, startup_camera_settings


class CameraStartupSettingsTest(unittest.TestCase):
    def test_dshow_startup_settings_reduce_default_overexposure(self):
        settings = startup_camera_settings("DSHOW")

        self.assertFalse(settings["auto_exposure"])
        self.assertLessEqual(settings["exposure"], -6)
        self.assertEqual(settings["gain"], 0)

    def test_any_backend_still_uses_low_exposure_defaults(self):
        settings = startup_camera_settings("ANY")

        self.assertFalse(settings["auto_exposure"])
        self.assertIn("exposure", settings)

    def test_exposure_tuning_candidates_cover_minus_3_to_minus_11(self):
        self.assertEqual(exposure_tuning_candidates(), [-3, -4, -5, -6, -7, -8, -9, -10, -11])

    def test_choose_best_exposure_prefers_high_focus_without_overexposure(self):
        samples = [
            {"exposure": -9, "focus_score": 20.0, "saturation_fraction": 0.0},
            {"exposure": -7, "focus_score": 90.0, "saturation_fraction": 0.02},
            {"exposure": -5, "focus_score": 120.0, "saturation_fraction": 0.40},
        ]

        selected = choose_best_exposure_result(samples)

        self.assertEqual(selected["exposure"], -7)


if __name__ == "__main__":
    unittest.main()
