"""OpenCV camera wrapper with backend and exposure controls."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local optional dependency
    cv2 = None


LOG = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised when the camera cannot be opened or read."""


BACKENDS = {
    "DSHOW": "CAP_DSHOW",
    "MSMF": "CAP_MSMF",
    "ANY": "CAP_ANY",
}

PROPERTY_NAMES = {
    "brightness": "CAP_PROP_BRIGHTNESS",
    "contrast": "CAP_PROP_CONTRAST",
    "saturation": "CAP_PROP_SATURATION",
    "gain": "CAP_PROP_GAIN",
    "exposure": "CAP_PROP_EXPOSURE",
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
    "auto_wb": "CAP_PROP_AUTO_WB",
    "wb_temperature": "CAP_PROP_WB_TEMPERATURE",
    "frame_width": "CAP_PROP_FRAME_WIDTH",
    "frame_height": "CAP_PROP_FRAME_HEIGHT",
    "fps": "CAP_PROP_FPS",
}


def _cv_property_id(name_or_id: str | int) -> int | None:
    if cv2 is None:
        return None
    if isinstance(name_or_id, int):
        return name_or_id
    constant_name = PROPERTY_NAMES.get(str(name_or_id).lower(), str(name_or_id))
    return getattr(cv2, constant_name, None)


def _backend_id(name: str) -> int:
    if cv2 is None:
        return 0
    constant = BACKENDS.get(name.upper(), "CAP_ANY")
    return int(getattr(cv2, constant, cv2.CAP_ANY))


def frame_brightness_diagnostics(frame) -> dict[str, float | bool]:
    if frame is None:
        return {
            "mean_brightness": 0.0,
            "saturation_fraction": 0.0,
            "underexposed_fraction": 0.0,
            "overexposed": False,
        }
    arr = np.asarray(frame)
    mean_brightness = float(arr.mean())
    saturation_fraction = float(np.mean(arr >= 250))
    underexposed_fraction = float(np.mean(arr <= 5))
    return {
        "mean_brightness": mean_brightness,
        "saturation_fraction": saturation_fraction,
        "underexposed_fraction": underexposed_fraction,
        "overexposed": saturation_fraction > 0.10 or mean_brightness > 220.0,
    }


class OpenCVCamera:
    def __init__(self) -> None:
        self._capture: Optional[Any] = None
        self.index: Optional[int] = None
        self.backend_name = "DSHOW"

    @property
    def is_open(self) -> bool:
        return bool(self._capture and self._capture.isOpened())

    def open(self, index: int, backend: str = "DSHOW") -> bool:
        self.close()
        self.index = int(index)
        self.backend_name = backend.upper()
        if cv2 is None:
            LOG.warning("OpenCV is not installed; entering no-camera mode")
            return False
        try:
            capture = cv2.VideoCapture(self.index, _backend_id(self.backend_name))
            if not capture.isOpened() and self.backend_name != "ANY":
                capture.release()
                capture = cv2.VideoCapture(self.index, cv2.CAP_ANY)
            if not capture.isOpened():
                capture.release()
                self._capture = None
                LOG.warning("camera index %s could not be opened; entering no-camera mode", self.index)
                return False
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture
            LOG.info("camera opened: index=%s backend=%s", self.index, self.backend_name)
            return True
        except Exception:
            LOG.exception("camera index %s failed to open", self.index)
            self._capture = None
            return False

    def close(self) -> None:
        if self._capture:
            try:
                self._capture.release()
            except Exception:
                LOG.exception("error while closing camera")
        self._capture = None

    def read_frame(self):
        if cv2 is None:
            raise CameraError("OpenCV is not installed")
        if not self.is_open:
            raise CameraError("camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("camera frame read failed")
        return frame

    def get_camera_properties(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name in PROPERTY_NAMES:
            prop_id = _cv_property_id(name)
            if not self.is_open or prop_id is None:
                result[name] = None
                continue
            try:
                value = float(self._capture.get(prop_id))
                result[name] = value
            except Exception:
                LOG.exception("camera property read failed: %s", name)
                result[name] = None
        LOG.info("camera properties: %s", result)
        return result

    def set_camera_property(self, name_or_id: str | int, value: float) -> float | None:
        prop_id = _cv_property_id(name_or_id)
        if not self.is_open or prop_id is None:
            LOG.warning("camera property unsupported or camera closed: %s", name_or_id)
            return None
        try:
            ok = bool(self._capture.set(prop_id, float(value)))
            readback = float(self._capture.get(prop_id))
            LOG.info("camera set property: %s target=%s ok=%s readback=%s", name_or_id, value, ok, readback)
            return readback
        except Exception:
            LOG.exception("camera property set failed: %s=%s", name_or_id, value)
            return None

    def set_auto_exposure(self, enabled: bool) -> float | None:
        candidates = [0.75, 1.0] if enabled else [0.25, 0.0]
        readback = None
        for value in candidates:
            readback = self.set_camera_property("auto_exposure", value)
            if readback is not None:
                LOG.info("auto exposure attempt enabled=%s value=%s readback=%s", enabled, value, readback)
        return readback

    def apply_camera_settings(self, settings: dict[str, float | bool | None]) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        if "auto_exposure" in settings and settings["auto_exposure"] is not None:
            result["auto_exposure"] = self.set_auto_exposure(bool(settings["auto_exposure"]))
        for name in ("exposure", "gain", "brightness", "contrast", "auto_wb", "wb_temperature"):
            if name in settings and settings[name] is not None:
                result[name] = self.set_camera_property(name, float(settings[name]))
        LOG.info("camera settings applied: %s", result)
        return result

    def reduce_overexposure(self) -> dict[str, float | bool | None]:
        result: dict[str, float | bool | None] = {
            "exposure": None,
            "gain": None,
            "saturation_fraction": None,
            "success": False,
        }
        if not self.is_open:
            LOG.warning("reduce overexposure requested while camera closed")
            return result
        LOG.info("reduce overexposure started")
        self.set_auto_exposure(False)
        for exposure in (-4, -5, -6, -7, -8, -9):
            result["exposure"] = exposure
            self.set_camera_property("exposure", exposure)
            saturation = self._read_saturation_fraction()
            result["saturation_fraction"] = saturation
            LOG.info("reduce overexposure exposure=%s saturation_fraction=%.4f", exposure, saturation)
            if saturation < 0.05:
                result["success"] = True
                return result
        result["gain"] = 0
        self.set_camera_property("gain", 0)
        saturation = self._read_saturation_fraction()
        result["saturation_fraction"] = saturation
        result["success"] = saturation < 0.05
        LOG.info("reduce overexposure final gain=0 saturation_fraction=%.4f success=%s", saturation, result["success"])
        return result

    def _read_saturation_fraction(self) -> float:
        try:
            frame = self.read_frame()
        except Exception:
            LOG.exception("camera read failed during overexposure reduction")
            return 1.0
        return float(frame_brightness_diagnostics(frame)["saturation_fraction"])
