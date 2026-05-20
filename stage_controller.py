"""Serial motor controller wrapper for the three-axis probe station."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

import serial

import stage_protocol as protocol


LOG = logging.getLogger(__name__)


class StageControllerError(Exception):
    """Raised for serial, protocol, or motion-control failures."""


def windows_device_path(port: str) -> str:
    """Return a Windows device path for COM ports, preserving non-COM names."""
    name = str(port).strip()
    upper = name.upper()
    if upper.startswith("\\\\.\\"):
        return name
    if upper.startswith("COM") and upper[3:].isdigit():
        return f"\\\\.\\{name}"
    return name


def is_windows_device_not_functioning_error(exc: BaseException) -> bool:
    text = str(exc)
    return "PermissionError(13" in text and (
        "device attached to the system is not functioning" in text.lower()
        or "None, 31" in text
    )


class StageController:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.25,
        write_timeout: float = 0.5,
        motion_timeout: float = 15.0,
        invert_x: bool = True,
        invert_y: bool = True,
        invert_z: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.motion_timeout = motion_timeout
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        self._serial: Optional[serial.Serial] = None
        self._rx_buffer = bytearray()
        self._operation_lock = threading.RLock()
        self._write_lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    def open(self) -> None:
        with self._operation_lock:
            if self.is_open:
                return
            errors: list[str] = []
            for label, port, write_timeout, low_control_lines in self._serial_open_attempts():
                try:
                    self._serial = self._open_serial_port(port, write_timeout, low_control_lines)
                    self._initialize_open_serial()
                    LOG.info("serial port opened using %s attempt on %s", label, port)
                    return
                except Exception as exc:
                    errors.append(f"{label} ({port}): {exc}")
                    self.close()
                    LOG.warning("serial open attempt failed: %s (%s): %s", label, port, exc)

            detail = "; ".join(errors)
            message = f"failed to open serial port {self.port}: {detail}"
            if any("PermissionError(13" in error and "31" in error for error in errors):
                message += (
                    ". Windows reports the device is present but failed while configuring it. "
                    "Try reconnecting the USB serial adapter, disabling/re-enabling the COM port in Device Manager, "
                    "or reinstalling the USB-serial driver if this persists."
                )
            raise StageControllerError(message)

    def _serial_open_attempts(self) -> list[tuple[str, str, float | None, bool]]:
        attempts: list[tuple[str, str, float | None, bool]] = [
            ("standard", self.port, self.write_timeout, False),
            ("standard-no-write-timeout", self.port, None, False),
        ]
        if sys.platform.startswith("win"):
            device_path = windows_device_path(self.port)
            if device_path != self.port:
                attempts.extend(
                    [
                        ("windows-device-path", device_path, self.write_timeout, False),
                        ("windows-device-path-no-write-timeout", device_path, None, False),
                    ]
                )
            attempts.extend(
                [
                    ("low-rts-dtr", self.port, self.write_timeout, True),
                    ("low-rts-dtr-no-write-timeout", self.port, None, True),
                ]
            )
            if device_path != self.port:
                attempts.extend(
                    [
                        ("windows-device-path-low-rts-dtr", device_path, self.write_timeout, True),
                        ("windows-device-path-low-rts-dtr-no-write-timeout", device_path, None, True),
                    ]
                )
        return attempts

    def _open_serial_port(
        self,
        port: str,
        write_timeout: float | None,
        low_control_lines: bool,
    ) -> serial.Serial:
        if low_control_lines:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = self.baudrate
            ser.bytesize = serial.EIGHTBITS
            ser.parity = serial.PARITY_NONE
            ser.stopbits = serial.STOPBITS_ONE
            ser.timeout = self.timeout
            ser.write_timeout = write_timeout
            ser.xonxoff = False
            ser.rtscts = False
            ser.dsrdtr = False
            ser.rts = False
            ser.dtr = False
            try:
                ser.open()
            except Exception as exc:
                try:
                    ser.close()
                except Exception:
                    pass
                raise exc
            return ser

        return serial.Serial(
            port=port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=write_timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def _initialize_open_serial(self) -> None:
        self._reset_serial_buffers()
        self._write(protocol.disable_realtime_position_upload())
        time.sleep(0.05)
        self._reset_serial_buffers()
        self._rx_buffer.clear()

    def _reset_serial_buffers(self) -> None:
        ser = self._require_serial()
        try:
            ser.reset_input_buffer()
        except Exception:
            LOG.exception("reset_input_buffer failed")
        try:
            ser.reset_output_buffer()
        except Exception:
            LOG.exception("reset_output_buffer failed")

    def close(self) -> None:
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                LOG.exception("error while closing serial port")
        self._serial = None
        self._rx_buffer.clear()

    def test_connection(self) -> bool:
        with self._operation_lock:
            frame = self._request(
                protocol.communication_test(),
                protocol.Command.TEST_CONNECTION_RESPONSE,
                expected_axis=0x00,
            )
            return frame.raw == bytes.fromhex("A3 AA 00 00 00 00 00 00 00 4D 0D 0A")

    def read_position(self, axis: str | protocol.Axis) -> int:
        with self._operation_lock:
            axis_value = protocol.normalize_axis(axis)
            if axis_value == protocol.Axis.ALL:
                raise StageControllerError("read_position expects X, Y, or Z; use read_all_positions for ALL")
            frame = self._request(protocol.read_position(axis_value), protocol.Command.READ_POSITION, axis_value)
            parsed = protocol.parse_position_payload(frame)
            return int(parsed["position"])

    def read_all_positions(self) -> dict[str, int]:
        with self._operation_lock:
            return {axis: self.read_position(axis) for axis in ("X", "Y", "Z")}

    def read_io(self) -> dict[str, int | str]:
        with self._operation_lock:
            frame = self._request(protocol.read_io(), protocol.Command.READ_IO, expected_axis=None)
            return protocol.parse_io_payload(frame)

    def move_relative(
        self,
        axis: str | protocol.Axis,
        logical_direction: int,
        pulses: int,
        speed_percent: int,
    ) -> None:
        with self._operation_lock:
            axis_value = protocol.normalize_axis(axis)
            if axis_value == protocol.Axis.ALL:
                raise StageControllerError("move_relative expects X, Y, or Z")
            if pulses <= 0:
                raise StageControllerError("pulses must be positive")

            direction = 1 if logical_direction >= 0 else -1
            if axis_value == protocol.Axis.X and self.invert_x:
                direction *= -1
            elif axis_value == protocol.Axis.Y and self.invert_y:
                direction *= -1
            elif axis_value == protocol.Axis.Z and self.invert_z:
                direction *= -1

            controller_direction = (
                protocol.Direction.POSITIVE if direction >= 0 else protocol.Direction.NEGATIVE
            )
            self._write(protocol.move_relative(axis_value, controller_direction, pulses, speed_percent))
            self._wait_for_arrival(axis_value)

    def stop_all(self) -> None:
        self._write(protocol.stop(protocol.Axis.ALL))

    def emergency_stop_all(self) -> None:
        self._write(protocol.emergency_stop(protocol.Axis.ALL))

    def _require_serial(self) -> serial.Serial:
        if not self._serial or not self._serial.is_open:
            raise StageControllerError("serial port is not open")
        return self._serial

    def _write(self, data: bytes) -> None:
        ser = self._require_serial()
        try:
            with self._write_lock:
                count = ser.write(data)
                ser.flush()
        except Exception as exc:
            raise StageControllerError(f"serial write failed: {exc}") from exc
        if count != len(data):
            raise StageControllerError(f"serial write incomplete: wrote {count} of {len(data)} bytes")

    def _read_frames_until(self, deadline: float) -> list[protocol.Frame]:
        ser = self._require_serial()
        frames: list[protocol.Frame] = []
        while time.monotonic() < deadline:
            waiting = ser.in_waiting
            chunk = ser.read(max(1, waiting))
            if chunk:
                self._rx_buffer.extend(chunk)
                try:
                    frames.extend(protocol.extract_frames(self._rx_buffer))
                except protocol.ProtocolError as exc:
                    self._rx_buffer.clear()
                    raise StageControllerError(f"invalid controller frame: {exc}") from exc
                if frames:
                    return frames
            else:
                time.sleep(0.01)
        return frames

    def _request(
        self,
        request_frame: bytes,
        expected_command: protocol.Command,
        expected_axis: protocol.Axis | int | None,
        timeout: float = 1.5,
    ) -> protocol.Frame:
        self._write(request_frame)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in self._read_frames_until(deadline):
                axis_matches = expected_axis is None or frame.axis == int(expected_axis)
                if frame.command == int(expected_command) and axis_matches:
                    return frame
                LOG.debug("ignored unexpected frame: command=0x%02X axis=%s", frame.command, frame.axis_name)
        axis_text = "any" if expected_axis is None else f"0x{int(expected_axis):02X}"
        raise StageControllerError(
            f"timeout waiting for command 0x{int(expected_command):02X} axis {axis_text}"
        )

    def _wait_for_arrival(self, axis: protocol.Axis) -> None:
        deadline = time.monotonic() + self.motion_timeout
        while time.monotonic() < deadline:
            for frame in self._read_frames_until(min(deadline, time.monotonic() + 0.25)):
                if protocol.is_arrival_feedback(frame, axis):
                    return
                LOG.debug("ignored frame while waiting for arrival: command=0x%02X", frame.command)

        LOG.warning("arrival feedback timed out for %s; trying position read confirmation", axis.name)
        try:
            self.read_position(axis)
        except Exception as exc:
            raise StageControllerError(f"motion did not report arrival and position confirmation failed: {exc}") from exc
