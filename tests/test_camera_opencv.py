import unittest

from camera_opencv import startup_camera_settings


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


if __name__ == "__main__":
    unittest.main()
