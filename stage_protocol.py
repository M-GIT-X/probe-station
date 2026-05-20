"""12-byte frame helpers for the four-axis controller custom protocol.

The physical controller supports X/Y/Z/A, but this project exposes only X/Y/Z
and ALL. A-axis commands are intentionally not generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


SEND_HEADER = 0x3A
RESPONSE_HEADER = 0xA3
FOOTER = b"\x0D\x0A"
FRAME_SIZE = 12


class Axis(IntEnum):
    X = 0x01
    Y = 0x02
    Z = 0x04
    ALL = 0xFF


class Direction(IntEnum):
    POSITIVE = 0x00
    NEGATIVE = 0x01


class StopMode(IntEnum):
    DECELERATE = 0x4A
    EMERGENCY = 0x49


class Command(IntEnum):
    TEST_CONNECTION = 0x55
    TEST_CONNECTION_RESPONSE = 0xAA
    READ_POSITION = 0xCB
    READ_IO = 0xD7
    MOVE_RELATIVE = 0xFA
    STOP = 0xFC
    MOVE_DONE = 0xB5
    DISABLE_REALTIME_POSITION_UPLOAD = 0xD4


class ProtocolError(Exception):
    """Raised when a controller frame is malformed or unsupported."""


@dataclass(frozen=True)
class Frame:
    command: int
    axis: int
    data: bytes
    raw: bytes

    @property
    def axis_name(self) -> str:
        try:
            return Axis(self.axis).name
        except ValueError:
            return f"0x{self.axis:02X}"


def checksum(data: bytes) -> int:
    return sum(data) & 0xFF


def hex_bytes(data: bytes) -> str:
    return data.hex(" ").upper()


def normalize_axis(axis: Axis | str) -> Axis:
    if isinstance(axis, Axis):
        if axis in (Axis.X, Axis.Y, Axis.Z, Axis.ALL):
            return axis
        raise ProtocolError(f"unsupported axis: {axis!r}")
    name = str(axis).strip().upper()
    try:
        return Axis[name]
    except KeyError as exc:
        raise ProtocolError(f"unsupported axis: {axis!r}; use X, Y, Z, or ALL") from exc


def make_12byte_frame(function_code: int, axis: int = 0x00, data: bytes = b"") -> bytes:
    if not 0 <= int(function_code) <= 0xFF:
        raise ProtocolError(f"function code out of range: {function_code!r}")
    if not 0 <= int(axis) <= 0xFF:
        raise ProtocolError(f"axis out of range: {axis!r}")
    if len(data) > 6:
        raise ProtocolError("data field must be at most 6 bytes")

    body = bytes([int(function_code), int(axis)]) + data.ljust(6, b"\x00")
    first_9 = bytes([SEND_HEADER]) + body
    return first_9 + bytes([checksum(first_9)]) + FOOTER


def parse_frame(frame: bytes) -> Frame:
    if len(frame) != FRAME_SIZE:
        raise ProtocolError(f"feedback frame must be 12 bytes, got {len(frame)}")
    if frame[0] != RESPONSE_HEADER:
        raise ProtocolError(f"bad feedback header: 0x{frame[0]:02X}")
    if frame[10:12] != FOOTER:
        raise ProtocolError(f"bad feedback footer: {frame.hex(' ')}")
    expected = checksum(frame[:9])
    if frame[9] != expected:
        raise ProtocolError(f"bad checksum: got 0x{frame[9]:02X}, expected 0x{expected:02X}")
    return Frame(command=frame[1], axis=frame[2], data=frame[3:9], raw=frame)


def extract_frames(buffer: bytearray) -> list[Frame]:
    """Extract complete 12-byte feedback frames from a byte buffer."""
    frames: list[Frame] = []
    while True:
        header_index = buffer.find(bytes([RESPONSE_HEADER]))
        if header_index < 0:
            buffer.clear()
            break
        if header_index:
            del buffer[:header_index]
        if len(buffer) < FRAME_SIZE:
            break
        raw = bytes(buffer[:FRAME_SIZE])
        del buffer[:FRAME_SIZE]
        frames.append(parse_frame(raw))
    return frames


def make_test_connection_command() -> bytes:
    return make_12byte_frame(Command.TEST_CONNECTION, axis=0x00)


def make_read_position_command(axis: Axis | str) -> bytes:
    return make_12byte_frame(Command.READ_POSITION, int(normalize_axis(axis)))


def make_read_io_command() -> bytes:
    return make_12byte_frame(Command.READ_IO, axis=0x00)


def make_move_relative_command(axis: Axis | str, direction: Direction | int, pulses: int, speed_percent: int) -> bytes:
    axis_value = normalize_axis(axis)
    if axis_value == Axis.ALL:
        raise ProtocolError("relative movement must target X, Y, or Z")
    if int(direction) not in (Direction.POSITIVE, Direction.NEGATIVE):
        raise ProtocolError("direction must be 0x00 positive or 0x01 negative")
    safe_pulses = max(0, int(pulses))
    if safe_pulses > 0xFFFFFFFF:
        raise ProtocolError("pulses exceed 32-bit range")
    safe_speed = max(1, min(100, int(speed_percent)))
    data = bytes([int(direction)]) + safe_pulses.to_bytes(4, "big") + bytes([safe_speed])
    return make_12byte_frame(Command.MOVE_RELATIVE, int(axis_value), data)


def make_stop_command(axis: Axis | str = Axis.ALL, emergency: bool = False) -> bytes:
    mode = StopMode.EMERGENCY if emergency else StopMode.DECELERATE
    return make_12byte_frame(Command.STOP, int(normalize_axis(axis)), bytes([mode]))


def make_disable_realtime_position_upload_command() -> bytes:
    return make_12byte_frame(Command.DISABLE_REALTIME_POSITION_UPLOAD, axis=0x00)


def validate_response_frame(frame: bytes) -> Frame:
    return parse_frame(frame)


def parse_position_response(frame: bytes | Frame) -> dict[str, int | bool | str]:
    parsed = frame if isinstance(frame, Frame) else parse_frame(frame)
    return parse_position_payload(parsed)


def parse_io_response(frame: bytes | Frame) -> dict[str, int | str]:
    parsed = frame if isinstance(frame, Frame) else parse_frame(frame)
    return parse_io_payload(parsed)


def communication_test() -> bytes:
    return make_test_connection_command()


def read_position(axis: Axis | str) -> bytes:
    return make_read_position_command(axis)


def read_io() -> bytes:
    return make_read_io_command()


def move_relative(axis: Axis | str, direction: Direction | int, pulses: int, speed_percent: int) -> bytes:
    return make_move_relative_command(axis, direction, pulses, speed_percent)


def stop(axis: Axis | str = Axis.ALL) -> bytes:
    return make_12byte_frame(Command.STOP, int(normalize_axis(axis)), bytes([StopMode.DECELERATE]))


def emergency_stop(axis: Axis | str = Axis.ALL) -> bytes:
    return make_12byte_frame(Command.STOP, int(normalize_axis(axis)), bytes([StopMode.EMERGENCY]))


def disable_realtime_position_upload() -> bytes:
    return make_disable_realtime_position_upload_command()


def parse_position_payload(frame: Frame) -> dict[str, int | bool | str]:
    if frame.command != int(Command.READ_POSITION):
        raise ProtocolError(f"not a position frame: 0x{frame.command:02X}")
    sign = frame.data[1]
    magnitude = int.from_bytes(frame.data[2:6], "big", signed=False)
    position = -magnitude if sign else magnitude
    return {
        "axis": Axis(frame.axis).name,
        "position": position,
        "running": bool(frame.data[0]),
    }


def parse_io_payload(frame: Frame) -> dict[str, int | str]:
    if frame.command != int(Command.READ_IO):
        raise ProtocolError(f"not an IO frame: 0x{frame.command:02X}")
    return {
        "home_bits": frame.axis,
        "limit_bits": int.from_bytes(frame.data[0:2], "big"),
        "inputs": frame.data[2],
        "outputs": frame.data[3],
        "raw_hex": frame.raw.hex(" "),
    }


def is_arrival_feedback(frame: Frame, axis: Axis | str | None = None) -> bool:
    if frame.command != int(Command.MOVE_DONE):
        return False
    if axis is None:
        return True
    return frame.axis == int(normalize_axis(axis))
