import unittest
from unittest.mock import patch

import stage_controller


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.is_open = False
        self.writes = []
        self.port = kwargs.get("port")
        self.baudrate = kwargs.get("baudrate")
        self.bytesize = kwargs.get("bytesize")
        self.parity = kwargs.get("parity")
        self.stopbits = kwargs.get("stopbits")
        self.timeout = kwargs.get("timeout")
        self.write_timeout = kwargs.get("write_timeout")
        self.xonxoff = kwargs.get("xonxoff", False)
        self.rtscts = kwargs.get("rtscts", False)
        self.dsrdtr = kwargs.get("dsrdtr", False)
        self.rts = True
        self.dtr = True

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        pass


class StageControllerOpenTest(unittest.TestCase):
    def test_windows_device_path_for_com_ports(self):
        self.assertEqual(stage_controller.windows_device_path("COM5"), "\\\\.\\COM5")
        self.assertEqual(stage_controller.windows_device_path("\\\\.\\COM5"), "\\\\.\\COM5")
        self.assertEqual(stage_controller.windows_device_path("/dev/cu.usbserial"), "/dev/cu.usbserial")

    def test_windows_open_falls_back_to_low_rts_dtr(self):
        created = []

        def serial_factory(*args, **kwargs):
            fake = FakeSerial(*args, **kwargs)
            created.append(fake)
            if args or kwargs:
                raise PermissionError(13, "A device attached to the system is not functioning.", None, 31)
            return fake

        controller = stage_controller.StageController(port="COM5")

        with patch.object(stage_controller.sys, "platform", "win32"), patch.object(
            stage_controller.serial, "Serial", side_effect=serial_factory
        ), patch.object(stage_controller.LOG, "warning"):
            controller.open()

        self.assertTrue(controller.is_open)
        self.assertFalse(controller._serial.rts)
        self.assertFalse(controller._serial.dtr)
        self.assertEqual(controller._serial.port, "COM5")
        self.assertGreaterEqual(len(created), 5)


if __name__ == "__main__":
    unittest.main()
