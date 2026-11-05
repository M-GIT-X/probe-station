"""Temporary, keyboard-only XYZ jog for a locally attached serial controller."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import signal
import sys
import threading

from stage_controller import StageController, StageControllerError


# Match the logical axes and signs of the main GUI's manual shortcuts.
_DIRECTIONS = {
    "w": ("Y", -1), "s": ("Y", 1),
    "a": ("X", 1), "d": ("X", -1),
    "q": ("Z", -1), "e": ("Z", 1),
    "down": ("Z", -1), "up": ("Z", 1),
}
_STOP_KEYS = {"space", "escape"}
_KEY_RELEASE_MS = 30
_POLL_MS = 30
_ARM_PROMPT = "Release the jog keys, then click this window to arm."


class JogWindow:
    def __init__(self, root, status, controller: StageController, pulses: int, speed: int) -> None:
        self.root = root
        self.status = status
        self.controller = controller
        self.pulses = pulses
        self.speed = speed
        self._workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xyz-jog")
        self._cancel = threading.Event()
        self._move: Future[None] | None = None
        self._stop: Future[None] | None = None
        self._stop_sent = False
        self._closing = False
        self._interrupted = False
        self._armed = False
        self._held: set[str] = set()
        self._releases: dict[str, str] = {}
        self.error = ""

        root.bind("<KeyPress>", self._key_press)
        root.bind("<KeyRelease>", self._key_release)
        root.bind("<Button-1>", lambda _event: self._set_armed(True))
        root.bind("<FocusOut>", lambda _event: self._set_armed(False))
        root.protocol("WM_DELETE_WINDOW", self._request_exit)

    def _set_armed(self, armed: bool) -> None:
        self._armed = armed
        if armed:
            self.root.focus_set()
        if self._move is None and not self._cancel.is_set() and not self._closing:
            self.status.set("Ready for one step at a time." if armed else _ARM_PROMPT)

    def _key_press(self, event) -> str | None:
        key = event.keysym.lower()
        if key == "x":
            self._request_exit()
            return "break"
        if key not in _DIRECTIONS and key not in _STOP_KEYS:
            return None

        release = self._releases.pop(key, None)
        if release is not None:
            self.root.after_cancel(release)
        if key in self._held:
            return "break"

        # Track presses dropped during motion too, so a held key cannot fire when it ends.
        self._held.add(key)
        if key in _STOP_KEYS:
            self._request_stop()
        elif self._armed and not self._closing and not self._cancel.is_set() and self._move is None:
            axis, direction = _DIRECTIONS[key]
            sign = "+" if direction > 0 else "−"
            self.status.set(f"{axis}{sign} step in progress…")
            self._move = self._workers.submit(
                self.controller.move_relative,
                axis, direction, self.pulses, self.speed,
                cancel=self._cancel, require_arrival_feedback=True,
            )
        return "break"

    def _key_release(self, event) -> None:
        key = event.keysym.lower()
        if key in self._held and key not in self._releases:
            # X11 can emit a release immediately followed by a press for an auto-repeat.
            self._releases[key] = self.root.after(_KEY_RELEASE_MS, self._release_key, key)

    def _release_key(self, key: str) -> None:
        self._releases.pop(key)
        self._held.discard(key)

    def _request_stop(self) -> None:
        self._cancel.set()
        if self._stop is None and not self._stop_sent:
            self.status.set("Sending software emergency stop…")
            self._stop = self._workers.submit(self.controller.emergency_stop_all)

    def _request_exit(self) -> None:
        self._closing = True
        if self._move is not None:
            self._request_stop()

    def _fail(self, operation: str, exc: Exception) -> None:
        self.error = f"{operation} failed: {exc}"
        self.status.set(self.error)
        print(self.error, file=sys.stderr)

    def _finish_stop(self, stop: Future[None]) -> None:
        try:
            stop.result()
        except Exception as exc:
            self._fail("Software emergency stop", exc)
        else:
            self._stop_sent = True
            detail = f"{self.error}\n" if self.error else ""
            self.status.set(f"{detail}Software emergency stop sent. Restart to jog again.")
        self._stop = None

    def _poll(self) -> None:
        if self._interrupted:
            self._request_exit()
        if self._move is not None and self._move.done():
            move, self._move = self._move, None
            if not self._cancel.is_set():
                try:
                    move.result()
                except Exception as exc:
                    self._fail("Jog step", exc)
                    self._request_stop()
                else:
                    next_step = "Ready for the next press." if self._armed else _ARM_PROMPT
                    self.status.set(f"Controller reported arrival. {next_step}")
        if self._stop is not None and self._stop.done():
            self._finish_stop(self._stop)

        if self._closing and self._move is None and self._stop is None:
            self.root.quit()
        else:
            self.root.after(_POLL_MS, self._poll)

    def _interrupt(self, _signal, _frame) -> None:
        self._interrupted = True

    def run(self) -> int:
        previous_handler = signal.signal(signal.SIGINT, self._interrupt)
        try:
            self.root.after(_POLL_MS, self._poll)
            self.root.mainloop()
        finally:
            try:
                self._request_exit()
                self._workers.shutdown(wait=True)
                if self._stop is not None:
                    self._finish_stop(self._stop)
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        if self._interrupted:
            return 130
        return int(bool(self.error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="local motor serial port (COM name or device path)")
    parser.add_argument("--step-pulses", type=int, default=1, help="pulses per step, 1–20 (default: 1)")
    parser.add_argument("--speed-percent", type=int, default=10, help="controller speed, 1–20%% (default: 10%%)")
    args = parser.parse_args(argv)
    port = args.port.strip()
    if not port:
        parser.error("--port must name the local motor serial port")
    if not 1 <= args.step_pulses <= 20:
        parser.error("--step-pulses must be between 1 and 20")
    if not 1 <= args.speed_percent <= 20:
        parser.error("--speed-percent must be between 1 and 20")

    try:
        import tkinter as tk
    except ImportError as exc:
        print(f"XYZ jog requires Tk and a graphical desktop: {exc}", file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"XYZ jog requires Tk and a graphical desktop: {exc}", file=sys.stderr)
        return 1

    root.withdraw()
    controller = StageController(
        port=port, motion_timeout=5.0, invert_x=False, invert_y=True, invert_z=False,
    )
    try:
        controller.open()
        if not controller.test_connection():
            raise StageControllerError("unexpected controller handshake response")

        root.title("XYZ jog")
        instructions = (
            f"{port} | {args.step_pulses} pulse(s) per press @ {args.speed_percent}%\n\n"
            "W / S: Y− / Y+     A / D: X+ / X−\n"
            "Q / E: Z− / Z+     Down / Up: Z− / Z+\n"
            "Space / Esc: software emergency stop     X: exit\n"
            "Keep the physical emergency stop within reach."
        )
        tk.Label(root, text=instructions, padx=16, pady=12).pack()
        status = tk.StringVar(root, _ARM_PROMPT)
        tk.Label(root, textvariable=status, wraplength=420, padx=16, pady=12).pack()
        window = JogWindow(root, status, controller, args.step_pulses, args.speed_percent)
        root.deiconify()
        return window.run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"XYZ jog failed: {exc}", file=sys.stderr)
        return 1
    finally:
        controller.close()
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
