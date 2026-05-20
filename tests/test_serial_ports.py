import unittest
from unittest.mock import patch

import app_gui


class SerialPortSelectionTest(unittest.TestCase):
    def test_windows_defaults_to_com5_when_present(self):
        with patch.object(app_gui.sys, "platform", "win32"), patch.object(
            app_gui, "list_serial_port_names", return_value=["COM3", "COM5"]
        ):
            self.assertEqual(app_gui.default_serial_port(), "COM5")

    def test_windows_defaults_to_detected_port_when_com5_is_missing(self):
        with patch.object(app_gui.sys, "platform", "win32"), patch.object(
            app_gui, "list_serial_port_names", return_value=["COM4", "COM3"]
        ):
            self.assertEqual(app_gui.default_serial_port(), "COM3")

    def test_windows_keeps_manual_default_when_no_ports_are_detected(self):
        with patch.object(app_gui.sys, "platform", "win32"), patch.object(
            app_gui, "list_serial_port_names", return_value=[]
        ):
            self.assertEqual(app_gui.default_serial_port(), "COM5")

    def test_windows_choices_include_detected_ports_and_common_defaults(self):
        with patch.object(app_gui.sys, "platform", "win32"), patch.object(
            app_gui, "list_serial_port_names", return_value=["COM11", "COM4"]
        ):
            choices = app_gui.serial_port_choices()

        self.assertEqual(choices[:2], ["COM4", "COM11"])
        self.assertIn("COM5", choices)

    def test_selected_port_is_rejected_only_when_ports_are_detected(self):
        with patch.object(app_gui, "list_serial_port_names", return_value=["COM3", "COM4"]):
            self.assertFalse(app_gui.selected_serial_port_is_listed("COM5"))

        with patch.object(app_gui, "list_serial_port_names", return_value=[]):
            self.assertTrue(app_gui.selected_serial_port_is_listed("COM5"))


if __name__ == "__main__":
    unittest.main()
