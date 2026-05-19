"""Small OpenCV camera wrapper."""

from __future__ import annotations

import logging
from typing import Optional

import cv2


LOG = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised when the camera cannot be opened or read."""


class OpenCVCamera:
    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self.index: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return bool(self._capture and self._capture.isOpened())

    def open(self, index: int) -> bool:
        self.close()
        self.index = int(index)
        try:
            capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                self._capture = None
                LOG.warning("camera index %s could not be opened; entering no-camera mode", self.index)
                return False
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture
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
        if not self.is_open:
            raise CameraError("camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("camera frame read failed")
        return frame
