"""Tkinter GUI for the three-axis automated probe station."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from camera_opencv import OpenCVCamera
from focus_metrics import auto_select_rois, calculate_focus_index
from stage_controller import StageController


LOG = logging.getLogger(__name__)


def list_serial_port_names() -> list[str]:
    try:
        from serial.tools import list_ports
    except Exception:
        LOG.exception("failed to import pyserial list_ports")
        return []
    return [port.device for port in list_ports.comports()]


def default_serial_port() -> str:
    if sys.platform.startswith("win"):
        return "COM5"
    return ""


def serial_port_choices() -> list[str]:
    ports = list_serial_port_names()
    if sys.platform.startswith("win"):
        defaults = [f"COM{i}" for i in range(3, 10)]
        return list(dict.fromkeys(ports + defaults))
    return ports


@dataclass
class ManualFocusAssistState:
    recording: bool = False
    best_focus_index: float | None = None
    best_z_abs: int | None = None
    best_z_rel: int | None = None
    best_timestamp: float | None = None

    def reset_best(self, keep_recording: bool) -> None:
        self.recording = keep_recording
        self.best_focus_index = None
        self.best_z_abs = None
        self.best_z_rel = None
        self.best_timestamp = None

    def record_sample(self, focus_index: float, z_abs: int, z_rel: int, timestamp: float | None = None) -> bool:
        if not self.recording:
            return False
        if timestamp is None:
            timestamp = time.time()
        if self.best_focus_index is None or self._is_meaningful_improvement(focus_index):
            self.best_focus_index = float(focus_index)
            self.best_z_abs = int(z_abs)
            self.best_z_rel = int(z_rel)
            self.best_timestamp = float(timestamp)
            return True
        return False

    def _is_meaningful_improvement(self, focus_index: float) -> bool:
        if self.best_focus_index is None:
            return True
        return focus_index > self.best_focus_index * 1.01 or focus_index > self.best_focus_index + 0.5


class ProbeStationApp(tk.Tk):
    def __init__(self, enable_focus_assist: bool = True, enable_autofocus: bool = False) -> None:
        super().__init__()
        self.title("Three-Axis Probe Station")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.enable_focus_assist = enable_focus_assist
        self.enable_autofocus = enable_autofocus
        self.controller: StageController | None = None
        self.camera = OpenCVCamera()
        self.device_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.absolute_pos = {"X": 0, "Y": 0, "Z": 0}
        self.position_available = {"X": False, "Y": False, "Z": False}
        self.software_origin = {"X": 0, "Y": 0, "Z": 0}
        self.focus_history: list[float] = []
        self.focus_z_history: list[int] = []
        self.focus_rois = None
        self.manual_focus_assist = ManualFocusAssistState()
        self._photo = None
        self._last_frame_time = 0.0
        self._position_poll_running = False

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(50, self._drain_device_queue)
        self.after(80, self._camera_loop)
        self.after(500, self._poll_positions)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.port_var = tk.StringVar(value=default_serial_port())
        self.camera_index_var = tk.StringVar(value="0")
        self.step_var = tk.StringVar(value="100")
        self.speed_var = tk.StringVar(value="10")
        self.status_var = tk.StringVar(value="Ready. 软件急停不能替代物理急停。")
        self.motor_status_var = tk.StringVar(value="Motor: disconnected")
        self.camera_status_var = tk.StringVar(value="Camera: not opened")
        self.focus_var = tk.StringVar(value="Focus index: 0.00")
        self.current_z_abs_var = tk.StringVar(value="Current Z absolute: waiting")
        self.current_z_rel_var = tk.StringVar(value="Current Z relative: waiting")
        self.best_focus_var = tk.StringVar(value="Best focus index: --")
        self.best_z_abs_var = tk.StringVar(value="Best Z absolute: --")
        self.best_z_rel_var = tk.StringVar(value="Best Z relative: --")
        self.best_focus_time_var = tk.StringVar(value="Best focus time: --")
        self.focus_recording_var = tk.StringVar(value="Recording: no")
        self.focus_assist_message_var = tk.StringVar(value="Open the camera, then start manual focus assist.")
        self.abs_pos_var = tk.StringVar(value="Abs X=0  Y=0  Z=0")
        self.rel_pos_var = tk.StringVar(value="Rel X=0  Y=0  Z=0")

        conn = ttk.LabelFrame(left, text="Devices", padding=8)
        conn.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        conn.columnconfigure(1, weight=1)
        ttk.Label(conn, text="Serial").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(
            conn,
            textvariable=self.port_var,
            values=serial_port_choices(),
            width=12,
        )
        self.port_combo.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(conn, text="Connect Motor", command=self._connect_motor).grid(row=0, column=2, padx=2)
        ttk.Button(conn, text="Refresh Ports", command=self._refresh_serial_ports).grid(row=0, column=3, padx=2)
        ttk.Label(conn, text="Camera").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(conn, textvariable=self.camera_index_var, width=12).grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(conn, text="Open / Switch", command=self._open_camera).grid(row=1, column=2, padx=2, pady=(6, 0))
        ttk.Label(conn, textvariable=self.motor_status_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(conn, textvariable=self.camera_status_var).grid(row=3, column=0, columnspan=3, sticky="w")

        move = ttk.LabelFrame(left, text="Manual Move", padding=8)
        move.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for col in range(3):
            move.columnconfigure(col, weight=1)
        ttk.Label(move, text="Step pulses").grid(row=0, column=0, sticky="w")
        ttk.Entry(move, textvariable=self.step_var, width=8).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(move, text="Speed %").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(move, textvariable=self.speed_var, width=8).grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))

        ttk.Button(move, text="Y+  W", command=lambda: self._move("Y", +1)).grid(row=2, column=1, sticky="ew", pady=(10, 2))
        ttk.Button(move, text="X-  A", command=lambda: self._move("X", -1)).grid(row=3, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(move, text="Stop Space", command=self._stop_all).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(move, text="X+  D", command=lambda: self._move("X", +1)).grid(row=3, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(move, text="Y-  S", command=lambda: self._move("Y", -1)).grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Button(move, text="Z-  Q", command=lambda: self._move("Z", -1)).grid(row=5, column=0, sticky="ew", padx=2, pady=(10, 2))
        ttk.Button(move, text="Z+  E", command=lambda: self._move("Z", +1)).grid(row=5, column=2, sticky="ew", padx=2, pady=(10, 2))
        ttk.Button(move, text="Software E-Stop  Esc / X", command=self._emergency_stop).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        ttk.Button(move, text="Set Current As Software Origin", command=self._set_software_origin).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )

        pos = ttk.LabelFrame(left, text="Position", padding=8)
        pos.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(pos, textvariable=self.abs_pos_var).grid(row=0, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.rel_pos_var).grid(row=1, column=0, sticky="w")

        if self.enable_focus_assist:
            focus = ttk.LabelFrame(left, text="手动辅助对焦 Manual Focus Assist", padding=8)
            focus.grid(row=3, column=0, sticky="ew", pady=(0, 8))
            focus.columnconfigure(0, weight=1)
            focus.columnconfigure(1, weight=1)
            ttk.Label(focus, textvariable=self.focus_assist_message_var, wraplength=300).grid(row=0, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.focus_var).grid(row=1, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.current_z_abs_var).grid(row=2, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.current_z_rel_var).grid(row=3, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.best_focus_var).grid(row=4, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.best_z_abs_var).grid(row=5, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.best_z_rel_var).grid(row=6, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.best_focus_time_var).grid(row=7, column=0, columnspan=2, sticky="w")
            ttk.Label(focus, textvariable=self.focus_recording_var).grid(row=8, column=0, columnspan=2, sticky="w")
            ttk.Button(focus, text="Start Manual Focus Assist", command=self._start_focus_assist).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 2))
            ttk.Button(focus, text="Stop Manual Focus Assist", command=self._stop_focus_assist).grid(row=10, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Button(focus, text="Go To Best Z", command=self._go_to_best_z).grid(row=11, column=0, sticky="ew", pady=2, padx=(0, 2))
            ttk.Button(focus, text="Reset Best Focus", command=self._reset_best_focus).grid(row=11, column=1, sticky="ew", pady=2, padx=(2, 0))
            self.curve_canvas = tk.Canvas(focus, width=280, height=90, bg="#101820", highlightthickness=0)
            self.curve_canvas.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        else:
            self.curve_canvas = None

        warning = ttk.Label(left, textvariable=self.status_var, wraplength=310, foreground="#8a4b00")
        warning.grid(row=4, column=0, sticky="ew")

        self.video_label = ttk.Label(right, text="No camera")
        self.video_label.grid(row=0, column=0, sticky="nsew")

    def _bind_keys(self) -> None:
        self.bind("<KeyPress-a>", lambda _e: self._move("X", -1))
        self.bind("<KeyPress-d>", lambda _e: self._move("X", +1))
        self.bind("<KeyPress-w>", lambda _e: self._move("Y", +1))
        self.bind("<KeyPress-s>", lambda _e: self._move("Y", -1))
        self.bind("<KeyPress-q>", lambda _e: self._move("Z", -1))
        self.bind("<KeyPress-e>", lambda _e: self._move("Z", +1))
        self.bind("<KeyPress-r>", lambda _e: self._move("Z", +1))
        self.bind("<KeyPress-f>", lambda _e: self._move("Z", -1))
        self.bind("<space>", lambda _e: self._stop_all())
        self.bind("<KeyPress-x>", lambda _e: self._emergency_stop())
        self.bind("<Escape>", lambda _e: self._emergency_stop())

    def _connect_motor(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            self._set_status("Serial port is empty. Click Refresh Ports or type the controller port.")
            return
        self._set_status(f"Opening motor controller on {port}...")

        def worker() -> None:
            controller = None
            try:
                controller = StageController(port=port)
                controller.open()
                controller.test_connection()
                positions = self._safe_read_positions(controller)
                self.device_queue.put(("motor_connected", (controller, positions)))
            except Exception as exc:
                if controller is not None:
                    controller.close()
                LOG.exception("motor connection failed")
                self.device_queue.put(("error", f"Motor connection failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_serial_ports(self) -> None:
        choices = serial_port_choices()
        self.port_combo.configure(values=choices)
        if self.port_var.get().strip() and self.port_var.get().strip() in choices:
            self._set_status("Serial ports refreshed.")
            return
        if sys.platform.startswith("win"):
            self.port_var.set("COM5")
        elif choices:
            self.port_var.set("")
        self._set_status(
            "Serial ports refreshed. Select the motor controller port before connecting."
        )

    def _open_camera(self) -> None:
        try:
            index = int(self.camera_index_var.get())
        except ValueError:
            self._set_status("Camera index must be an integer.")
            return
        ok = self.camera.open(index)
        if ok:
            self.focus_rois = None
            self.camera_status_var.set(f"Camera: opened index {index}")
            self._set_status("Camera opened.")
        else:
            self.camera_status_var.set("Camera: no-camera mode")
            self.video_label.configure(text="No camera", image="")
            self._set_status("Camera could not be opened; GUI remains available.")

    def _move(self, axis: str, direction: int) -> None:
        if not self.controller or not self.controller.is_open:
            self._set_status("Motor is not connected.")
            return
        try:
            pulses = max(1, int(float(self.step_var.get())))
            speed = max(1, min(100, int(float(self.speed_var.get()))))
        except ValueError:
            self._set_status("Step and speed must be numbers.")
            return

        def worker() -> None:
            try:
                self.controller.move_relative(axis, direction, pulses, speed)
                positions = self._safe_read_positions(self.controller)
                self.device_queue.put(("positions", positions))
            except Exception as exc:
                LOG.exception("manual move failed")
                self.device_queue.put(("error", f"Move {axis}{'+' if direction > 0 else '-'} failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_all(self) -> None:
        if not self.controller or not self.controller.is_open:
            return
        self._set_status("Sending controlled stop...")
        threading.Thread(target=self._safe_stop_worker, daemon=True).start()

    def _safe_stop_worker(self) -> None:
        try:
            self.controller.stop_all()
            self.device_queue.put(("status", "Controlled stop sent."))
        except Exception as exc:
            LOG.exception("controlled stop failed")
            self.device_queue.put(("error", f"Controlled stop failed: {exc}"))

    def _emergency_stop(self) -> None:
        self._set_status("SOFTWARE emergency stop sent. 软件急停不能替代物理急停。")
        if not self.controller or not self.controller.is_open:
            return

        def worker() -> None:
            try:
                self.controller.emergency_stop_all()
            except Exception as exc:
                LOG.exception("software emergency stop failed")
                self.device_queue.put(("error", f"Software emergency stop failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_software_origin(self) -> None:
        self.software_origin = dict(self.absolute_pos)
        self._update_position_labels()
        self._set_status("Software origin set locally. No D3 clear command was sent.")

    def _start_focus_assist(self) -> None:
        self.manual_focus_assist.reset_best(keep_recording=True)
        self.focus_history.clear()
        self.focus_z_history.clear()
        self._draw_focus_curve()
        self._refresh_focus_assist_labels()
        if not self.camera.is_open:
            LOG.warning("camera not opened while starting manual focus assist")
            self.focus_assist_message_var.set("相机未打开，无法进行 focus assist")
        self._set_status("Manual focus assist started. Move Z manually to collect focus samples.")

    def _stop_focus_assist(self) -> None:
        self.manual_focus_assist.recording = False
        self._refresh_focus_assist_labels()
        self._set_status("Manual focus assist stopped.")

    def _reset_best_focus(self) -> None:
        self.manual_focus_assist.reset_best(keep_recording=self.manual_focus_assist.recording)
        self.focus_history.clear()
        self.focus_z_history.clear()
        self._draw_focus_curve()
        self._refresh_focus_assist_labels()
        self._set_status("Manual focus best record reset.")

    def _go_to_best_z(self) -> None:
        best_z_abs = self.manual_focus_assist.best_z_abs
        if best_z_abs is None:
            self._set_status("No best Z recorded yet.")
            return
        if not self.controller or not self.controller.is_open:
            self._set_status("Motor is not connected.")
            return
        if not self.position_available["Z"]:
            LOG.error("Z position unavailable; cannot go to best Z")
            self._set_status("Current Z position is unavailable; cannot Go To Best Z.")
            return
        confirmed = messagebox.askokcancel(
            "Confirm Go To Best Z",
            "即将移动 Z 轴回到记录的最佳焦点位置。请确认样品/探针/镜头安全，物理急停可用。\n\n"
            "软件急停不能替代物理急停。",
        )
        if not confirmed:
            self._set_status("Go To Best Z cancelled.")
            return
        delta = best_z_abs - self.absolute_pos["Z"]
        if delta == 0:
            self._set_status("Already at recorded best Z.")
            return
        direction = 1 if delta > 0 else -1
        pulses = abs(delta)
        speed = min(self._read_speed(default=2), 5)
        LOG.info("Go To Best Z started: current=%s best=%s delta=%s speed=%s", self.absolute_pos["Z"], best_z_abs, delta, speed)

        def worker() -> None:
            try:
                self.controller.move_relative("Z", direction, pulses, speed)
                positions = self._safe_read_positions(self.controller)
                self.device_queue.put(("positions", positions))
                LOG.info("Go To Best Z completed")
                self.device_queue.put(("status", "Moved to recorded best Z."))
            except Exception as exc:
                LOG.exception("go to best z failed")
                self.device_queue.put(("error", f"Go To Best Z failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _read_speed(self, default: int = 10) -> int:
        try:
            return max(1, min(100, int(float(self.speed_var.get()))))
        except ValueError:
            return default

    def _safe_read_positions(self, controller: StageController) -> dict[str, int]:
        try:
            return controller.read_all_positions()
        except Exception:
            LOG.exception("read_all_positions failed; trying per-axis reads")
            return {axis: controller.read_position(axis) for axis in ("X", "Y", "Z")}

    def _poll_positions(self) -> None:
        if self.controller and self.controller.is_open and not self._position_poll_running:
            self._position_poll_running = True

            def worker() -> None:
                try:
                    positions = self._safe_read_positions(self.controller)
                    self.device_queue.put(("positions", positions))
                except Exception:
                    LOG.exception("position poll failed")
                finally:
                    self.device_queue.put(("position_poll_done", None))

            threading.Thread(target=worker, daemon=True).start()
        self.after(700, self._poll_positions)

    def _camera_loop(self) -> None:
        if self.camera.is_open:
            try:
                frame = self.camera.read_frame()
                if self.focus_rois is None:
                    self.focus_rois = auto_select_rois(frame)
                focus_index = calculate_focus_index(frame, self.focus_rois)
                self._update_focus(focus_index)
                self._show_frame(frame)
            except Exception:
                LOG.exception("camera loop error")
                self.camera_status_var.set("Camera: read error, no-camera mode")
                self.camera.close()
        self.after(60, self._camera_loop)

    def _update_focus(self, focus_index: float) -> None:
        self.focus_var.set(f"Focus index: {focus_index:.2f}")
        if not self.enable_focus_assist:
            return
        if not self.manual_focus_assist.recording:
            self._refresh_focus_assist_labels()
            return
        if not self.camera.is_open:
            LOG.warning("camera not opened while trying to record focus")
            self.focus_assist_message_var.set("相机未打开，无法进行 focus assist")
            return
        if not self.position_available["Z"]:
            LOG.warning("Z position unavailable while trying to record focus")
            self.focus_assist_message_var.set("等待 Z 位置")
            self._refresh_focus_assist_labels()
            return
        z = self.absolute_pos["Z"]
        z_rel = z - self.software_origin["Z"]
        self.focus_history.append(float(focus_index))
        self.focus_z_history.append(z)
        if len(self.focus_history) > 240:
            self.focus_history = self.focus_history[-240:]
            self.focus_z_history = self.focus_z_history[-240:]
        if self.manual_focus_assist.record_sample(float(focus_index), z_abs=z, z_rel=z_rel):
            LOG.info(
                "best focus updated: focus=%.2f z_abs=%s z_rel=%s",
                self.manual_focus_assist.best_focus_index,
                self.manual_focus_assist.best_z_abs,
                self.manual_focus_assist.best_z_rel,
            )
        self.focus_assist_message_var.set("Recording manual focus samples.")
        self._refresh_focus_assist_labels()
        self._draw_focus_curve()

    def _show_frame(self, frame) -> None:
        try:
            import cv2
        except ImportError:
            self.camera_status_var.set("Camera: OpenCV is not installed")
            self.camera.close()
            return

        now = time.monotonic()
        if now - self._last_frame_time < 0.04:
            return
        self._last_frame_time = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_w = max(320, self.video_label.winfo_width())
        label_h = max(240, self.video_label.winfo_height())
        height, width = rgb.shape[:2]
        scale = min(label_w / width, label_h / height)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        header = f"P6\n{new_w} {new_h}\n255\n".encode("ascii")
        ppm = header + np.ascontiguousarray(resized).tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        self.video_label.configure(image=self._photo, text="")

    def _draw_focus_curve(self) -> None:
        canvas = self.curve_canvas
        if canvas is None:
            return
        canvas.delete("all")
        width = max(10, canvas.winfo_width())
        height = max(10, canvas.winfo_height())
        canvas.create_rectangle(0, 0, width, height, fill="#101820", outline="")
        if self.manual_focus_assist.recording and self.manual_focus_assist.best_focus_index is not None:
            canvas.create_text(
                8,
                8,
                anchor="nw",
                fill="#e6edf3",
                text=f"Best {self.manual_focus_assist.best_focus_index:.2f}",
            )
        if len(self.focus_history) < 2:
            return
        values = np.asarray(self.focus_history, dtype=np.float64)
        lo = float(values.min())
        hi = float(values.max())
        span = max(1.0, hi - lo)
        points = []
        for index, value in enumerate(values):
            x = index * (width - 1) / max(1, len(values) - 1)
            y = height - 4 - ((value - lo) / span) * (height - 8)
            points.extend([x, y])
        canvas.create_line(*points, fill="#4cc9f0", width=2)

    def _drain_device_queue(self) -> None:
        try:
            while True:
                kind, payload = self.device_queue.get_nowait()
                if kind == "motor_connected":
                    controller, positions = payload
                    if self.controller and self.controller is not controller:
                        self.controller.close()
                    self.controller = controller
                    self.motor_status_var.set(f"Motor: connected {controller.port}")
                    if positions:
                        self.absolute_pos.update(positions)
                        self._mark_positions_available(positions)
                        self._update_position_labels()
                    self._set_status("Motor connected. D4 realtime upload disable was sent.")
                elif kind == "positions":
                    self.absolute_pos.update(payload)
                    self._mark_positions_available(payload)
                    self._update_position_labels()
                elif kind == "position_poll_done":
                    self._position_poll_running = False
                elif kind == "status":
                    self._set_status(str(payload))
                elif kind == "error":
                    self._set_status(str(payload))
        except queue.Empty:
            pass
        self.after(50, self._drain_device_queue)

    def _update_position_labels(self) -> None:
        rel = {axis: self.absolute_pos[axis] - self.software_origin[axis] for axis in ("X", "Y", "Z")}
        self.abs_pos_var.set(
            f"Abs X={self.absolute_pos['X']}  Y={self.absolute_pos['Y']}  Z={self.absolute_pos['Z']}"
        )
        self.rel_pos_var.set(f"Rel X={rel['X']}  Y={rel['Y']}  Z={rel['Z']}")
        self._refresh_focus_assist_labels()

    def _mark_positions_available(self, positions: dict[str, int]) -> None:
        for axis in positions:
            if axis in self.position_available:
                self.position_available[axis] = True

    def _refresh_focus_assist_labels(self) -> None:
        if not self.enable_focus_assist:
            return
        state = self.manual_focus_assist
        if self.position_available["Z"]:
            z_abs = self.absolute_pos["Z"]
            z_rel = z_abs - self.software_origin["Z"]
            self.current_z_abs_var.set(f"Current Z absolute: {z_abs}")
            self.current_z_rel_var.set(f"Current Z relative: {z_rel}")
        else:
            self.current_z_abs_var.set("Current Z absolute: waiting")
            self.current_z_rel_var.set("Current Z relative: waiting")

        self.focus_recording_var.set(f"Recording: {'yes' if state.recording else 'no'}")
        if state.best_focus_index is None:
            self.best_focus_var.set("Best focus index: --")
            self.best_z_abs_var.set("Best Z absolute: --")
            self.best_z_rel_var.set("Best Z relative: --")
            self.best_focus_time_var.set("Best focus time: --")
            return

        self.best_focus_var.set(f"Best focus index: {state.best_focus_index:.2f}")
        self.best_z_abs_var.set(f"Best Z absolute: {state.best_z_abs}")
        self.best_z_rel_var.set(f"Best Z relative: {state.best_z_rel}")
        if state.best_timestamp is None:
            self.best_focus_time_var.set("Best focus time: --")
        else:
            time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state.best_timestamp))
            self.best_focus_time_var.set(f"Best focus time: {time_text}")

    def _set_status(self, message: str) -> None:
        LOG.info(message)
        self.status_var.set(message)

    def _on_close(self) -> None:
        try:
            if self.controller and self.controller.is_open:
                self.controller.stop_all()
        except Exception:
            LOG.exception("stop on close failed")
        try:
            if self.controller:
                self.controller.close()
            self.camera.close()
        finally:
            self.destroy()
