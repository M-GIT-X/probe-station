"""Bounded Z moves checked against controller arrival and pulse-counter feedback."""

from __future__ import annotations

import threading

from stage_controller import StageController, StageControllerError


class ZMotion:
    def __init__(self, controller: StageController, lower: int, upper: int, speed: int):
        if type(lower) is not int or type(upper) is not int or lower >= upper:
            raise ValueError("Z bounds must be integer pulse counts with lower < upper")
        if type(speed) is not int or not 1 <= speed <= 20:
            raise ValueError("Autofocus speed must be between 1 and 20 percent")
        if controller.invert_z:
            raise ValueError("ZMotion requires the trial controller's invert_z=False mapping")
        self.controller = controller
        self.lower = lower
        self.upper = upper
        self.speed = speed
        self._expected: int | None = None

    def _read_idle(self) -> int:
        state = self.controller.read_axis_state("Z")
        if state["running"]:
            raise StageControllerError("Z controller still reports motion")
        position = int(state["position"])
        if not self.lower <= position <= self.upper:
            raise StageControllerError("Z pulse counter is outside the approved autofocus bounds")
        return position

    def position(self) -> int:
        position = self._read_idle()
        if self._expected is not None and position != self._expected:
            raise StageControllerError("Z pulse counter changed outside the autofocus move sequence")
        self._expected = position
        return position

    def move_to(self, target: int, cancel: threading.Event) -> None:
        if cancel.is_set():
            raise InterruptedError("Autofocus canceled")
        if type(target) is not int or not self.lower <= target <= self.upper:
            raise ValueError("Autofocus target is outside the approved Z bounds")
        current = self.position()
        if cancel.is_set():
            raise InterruptedError("Autofocus canceled")
        delta = target - current
        if delta:
            self.controller.move_relative(
                "Z", 1 if delta > 0 else -1, abs(delta), self.speed,
                cancel=cancel, require_arrival_feedback=True,
            )
        if cancel.is_set():
            raise InterruptedError("Autofocus canceled")
        actual = self._read_idle()
        if actual != target:
            raise StageControllerError(f"Z pulse counter {actual} does not match target {target}")
        self._expected = target

    def stop(self) -> None:
        self.controller.emergency_stop_all()
