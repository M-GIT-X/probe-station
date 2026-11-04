"""One camera reader, bounded latest-frame delivery, and cancellable bursts."""

from __future__ import annotations

import threading
import time
import math
from typing import Callable

import numpy as np

from camera_opencv import OpenCVCamera
from .models import CapturedFrame


class CaptureError(RuntimeError):
    pass


class FrameStream:
    """Own a camera in one worker; callers never read VideoCapture themselves.

    Sequence numbers count successful host reads, not sensor exposures. Frames
    may be skipped by slow consumers, but are never returned twice to a burst.
    Receipt times alone do not prove a frame was exposed after a stage move.
    """

    def __init__(self, camera_factory: Callable = OpenCVCamera) -> None:
        self._camera_factory = camera_factory
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: CapturedFrame | None = None
        self._error: str | None = None
        self.properties: dict = {}
        self.source: dict = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active(self) -> bool:
        return self.running and not self._stop.is_set() and self.error is None

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._error

    def start(self, index: int, backend: str = "DSHOW", settings: dict | None = None) -> None:
        if self.running:
            raise CaptureError("Camera reader is already running")
        with self._condition:
            self._latest = None
            self._error = None
            self.properties = {}
            self.source = {"camera_index": index, "requested_backend": backend,
                           "timestamp_source": "host_read_interval",
                           "sensor_timestamps_available": False}
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(index, backend, settings), daemon=True)
        self._thread.start()

    def _run(self, index: int, backend: str, settings: dict | None) -> None:
        camera = None
        try:
            camera = self._camera_factory()
            if not camera.open(index, backend):
                raise CaptureError("Cannot open camera; close other apps using it")
            if settings:
                camera.apply_camera_settings(settings)
                self.source["requested_settings"] = dict(settings)
            properties = camera.get_camera_properties()
            with self._condition:
                self.properties = {
                    key: float(value) if value is not None and np.isfinite(value) else None
                    for key, value in properties.items()
                }
            sequence = 0
            while not self._stop.is_set():
                started = time.monotonic()
                raw = camera.read_frame()
                received = time.monotonic()
                frame = np.array(raw, copy=True)
                if frame.dtype != np.uint8 or frame.ndim not in (2, 3) or frame.size == 0:
                    raise CaptureError("Camera must return nonempty uint8 grayscale or BGR images")
                if frame.ndim == 3 and frame.shape[2] != 3:
                    raise CaptureError("Camera returned an unsupported channel count")
                frame.flags.writeable = False
                sequence += 1
                with self._condition:
                    self._latest = CapturedFrame(sequence, started, received, frame)
                    self._condition.notify_all()
                # Prevent a misbehaving backend/fake camera from busy-spinning.
                self._stop.wait(0.001)
        except Exception as exc:
            with self._condition:
                self._error = str(exc)
                self._condition.notify_all()
        finally:
            if camera is not None:
                camera.close()
            self._stop.set()
            with self._condition:
                self._condition.notify_all()

    def latest(self) -> CapturedFrame | None:
        with self._condition:
            return self._latest

    def wait_next(self, after_sequence: int, *, timeout: float = 2.0,
                  cancel: threading.Event | None = None) -> CapturedFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("Capture cancelled")
                if self._error:
                    raise CaptureError(self._error)
                if self._stop.is_set():
                    raise CaptureError("Camera reader stopped")
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CaptureError("Timed out waiting for a new camera read")
                self._condition.wait(min(remaining, 0.05))

    def collect(self, count: int, *, timeout: float = 8.0, discard: int = 3,
                cancel: threading.Event | None = None) -> tuple[CapturedFrame, ...]:
        """Collect reads begun after this request, discarding a few first reads.

        Discarding is a precaution, not a calibration of camera buffering.
        The operator must keep Z and the scene still for the entire burst.
        """
        if type(count) is not int or type(discard) is not int or not 3 <= count <= 30 or not 0 <= discard <= 30:
            raise ValueError("Burst needs 3–30 frames and 0–30 discarded reads")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Capture timeout must be finite and positive")
        started = time.monotonic()
        deadline = started + timeout
        latest = self.latest()
        sequence = latest.sequence if latest is not None else 0
        frames: list[CapturedFrame] = []
        while len(frames) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CaptureError("Timed out collecting the requested burst")
            frame = self.wait_next(sequence, timeout=remaining, cancel=cancel)
            sequence = frame.sequence
            if frame.read_started < started:
                continue
            if discard:
                discard -= 1
                continue
            frames.append(frame)
        return tuple(frames)

    def stop(self, timeout: float = 2.0) -> bool:
        """Signal then join; never release a capture concurrently with read()."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.running
