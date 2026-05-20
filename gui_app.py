"""Tkinter GUI for the three-axis automated probe station."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import sys
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from camera_opencv import OpenCVCamera
from focus_metrics import auto_select_rois, brightness_diagnostics, calculate_focus_index, robust_representative
from stage_controller import StageController


LOG = logging.getLogger(__name__)

APP_TITLE = "Three-Axis Probe Station"


class Mode(Enum):
    MANUAL = "Manual Mode"
    FOCUS_ASSIST = "Manual Focus Assist"
    AUTO_FOCUS = "Auto Focus"


INVERT_X_DIRECTION = True
INVERT_Y_DIRECTION = True
INVERT_Z_DIRECTION = False

SAFE_MODE = True
SAFE_MAX_MANUAL_STEP = 50
SAFE_MAX_MANUAL_SPEED = 5
SAFE_MAX_AUTOFOCUS_RANGE = 100
SAFE_MAX_AUTOFOCUS_STEP = 20
SAFE_MAX_AUTOFOCUS_SPEED = 5
MIN_SAMPLE_FRAMES = 3


def list_serial_port_names() -> list[str]:
    try:
        from serial.tools import list_ports
    except Exception:
        LOG.exception("failed to import pyserial list_ports")
        return []
    return [port.device for port in list_ports.comports()]


def _serial_port_sort_key(port: str) -> tuple[int, str]:
    name = port.upper()
    if name.startswith("COM"):
        suffix = name[3:]
        if suffix.isdigit():
            return int(suffix), name
    return 10_000, name


def default_serial_port() -> str:
    ports = sorted(list_serial_port_names(), key=_serial_port_sort_key)
    if sys.platform.startswith("win"):
        if "COM5" in {port.upper() for port in ports}:
            return "COM5"
        if ports:
            return ports[0]
        return "COM5"
    return ""


def serial_port_choices() -> list[str]:
    ports = sorted(list_serial_port_names(), key=_serial_port_sort_key)
    if sys.platform.startswith("win"):
        defaults = [f"COM{i}" for i in range(3, 10)]
        return list(dict.fromkeys(ports + defaults))
    return ports


def selected_serial_port_is_listed(port: str) -> bool:
    detected = {name.upper() for name in list_serial_port_names()}
    if not detected:
        return True
    return port.upper() in detected


def app_stage_title(enable_focus_assist: bool, enable_autofocus: bool) -> str:
    return APP_TITLE


def should_ignore_axis_shortcut(widget) -> bool:
    if widget is None:
        return False
    try:
        widget_class = widget.winfo_class()
    except Exception:
        return False
    return widget_class in {"Entry", "TEntry", "Combobox", "TCombobox", "Spinbox", "TSpinbox"}


def logical_direction_to_controller_direction(axis: str, logical_sign: int) -> int:
    sign = 1 if logical_sign >= 0 else -1
    invert_by_axis = {
        "X": INVERT_X_DIRECTION,
        "Y": INVERT_Y_DIRECTION,
        "Z": INVERT_Z_DIRECTION,
    }
    if invert_by_axis.get(axis.upper(), False):
        sign *= -1
    return sign


def clamp_manual_motion_params(pulses: int, speed: int) -> tuple[int, int, bool]:
    pulses = max(1, int(pulses))
    speed = max(1, min(100, int(speed)))
    if not SAFE_MODE:
        return pulses, speed, False
    clamped_pulses = min(pulses, SAFE_MAX_MANUAL_STEP)
    clamped_speed = min(speed, SAFE_MAX_MANUAL_SPEED)
    return clamped_pulses, clamped_speed, (clamped_pulses, clamped_speed) != (pulses, speed)


@dataclass(frozen=True)
class AutofocusParams:
    scan_range: int = 20
    scan_step: int = 5
    autofocus_speed: int = 2
    settle_seconds: float = 0.5
    sample_seconds: float = 1.5
    near_best_ratio: float = 0.96


@dataclass(frozen=True)
class AutofocusSamplePoint:
    offset: int
    score: float
    iqr: float
    frame_count: int


@dataclass
class AutofocusRunState:
    running: bool = False
    stop_requested: bool = False
    current_offset: int = 0
    current_score: float | None = None
    best_score: float | None = None
    best_offset: int | None = None
    final_offset: int | None = None
    confirm_score: float | None = None
    confirm_iqr: float | None = None
    sample_points: list[AutofocusSamplePoint] | None = None

    def reset(self) -> None:
        self.running = False
        self.stop_requested = False
        self.current_offset = 0
        self.current_score = None
        self.best_score = None
        self.best_offset = None
        self.final_offset = None
        self.confirm_score = None
        self.confirm_iqr = None
        self.sample_points = []


def clamp_autofocus_params(params: AutofocusParams) -> tuple[AutofocusParams, bool]:
    if not SAFE_MODE:
        return params, False
    clamped = AutofocusParams(
        scan_range=min(params.scan_range, SAFE_MAX_AUTOFOCUS_RANGE),
        scan_step=min(params.scan_step, SAFE_MAX_AUTOFOCUS_STEP),
        autofocus_speed=min(params.autofocus_speed, SAFE_MAX_AUTOFOCUS_SPEED),
        settle_seconds=params.settle_seconds,
        sample_seconds=params.sample_seconds,
        near_best_ratio=params.near_best_ratio,
    )
    return clamped, clamped != params


def build_scan_offsets(scan_range: int, scan_step: int) -> list[int]:
    scan_range = max(1, int(scan_range))
    scan_step = max(1, int(scan_step))
    offsets = list(range(-scan_range, scan_range + 1, scan_step))
    if offsets[-1] != scan_range:
        offsets.append(scan_range)
    return offsets


def select_final_autofocus_point(
    points: list[AutofocusSamplePoint],
    near_best_ratio: float,
) -> AutofocusSamplePoint:
    if not points:
        raise ValueError("no autofocus points")
    peak_score = max(point.score for point in points)
    threshold = peak_score * near_best_ratio
    candidates = [point for point in points if point.score >= threshold]
    return min(
        candidates,
        key=lambda point: (
            round(point.iqr / max(point.score, 1.0), 3),
            abs(point.offset),
            -point.score,
        ),
    )


def _interquartile_range(values: list[float]) -> float:
    if not values:
        return 0.0
    data = np.asarray(values, dtype=np.float64)
    q75, q25 = np.percentile(data, [75, 25])
    return float(q75 - q25)


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
    def __init__(self, enable_focus_assist: bool = True, enable_autofocus: bool = True) -> None:
        super().__init__()
        self.app_stage_title = APP_TITLE
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1080, 700)

        self.enable_focus_assist = True
        self.enable_autofocus = True
        self.controller: StageController | None = None
        self.camera = OpenCVCamera()
        self.camera_lock = threading.Lock()
        self.device_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.absolute_pos = {"X": 0, "Y": 0, "Z": 0}
        self.position_available = {"X": False, "Y": False, "Z": False}
        self.software_origin = {"X": 0, "Y": 0, "Z": 0}
        self.focus_history: list[float] = []
        self.focus_z_history: list[int] = []
        self.focus_rois = None
        self.last_focus_info = {
            "mean_brightness": 0.0,
            "frame_saturation": 0.0,
            "underexposed_fraction": 0.0,
            "overexposed": False,
        }
        self.manual_focus_assist = ManualFocusAssistState()
        self.autofocus = AutofocusRunState()
        self.autofocus.reset()
        self._photo = None
        self._last_frame_time = 0.0
        self._position_poll_running = False
        self._closing = False
        self._after_ids: set[str] = set()

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._schedule_after(50, self._drain_device_queue)
        self._schedule_after(80, self._camera_loop)
        self._schedule_after(500, self._poll_positions)

    def _schedule_after(self, delay_ms: int, callback) -> None:
        if self._closing:
            return

        after_id = ""

        def wrapped_callback():
            self._after_ids.discard(after_id)
            if not self._closing:
                callback()

        after_id = self.after(delay_ms, wrapped_callback)
        self._after_ids.add(after_id)

    def _cancel_after_callbacks(self) -> None:
        self._closing = True
        for after_id in list(self._after_ids):
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass
            finally:
                self._after_ids.discard(after_id)

    def destroy(self) -> None:
        self._cancel_after_callbacks()
        super().destroy()

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
        self.camera_backend_var = tk.StringVar(value="DSHOW")
        self.mode_var = tk.StringVar(value=Mode.MANUAL.value)
        self.current_mode_var = tk.StringVar(value=f"Current mode: {Mode.MANUAL.value}")
        self.mode_message_var = tk.StringVar(value="Manual Mode: manual X/Y/Z control only; no recording and no automatic movement.")
        self.step_var = tk.StringVar(value="10")
        self.speed_var = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value=f"{APP_TITLE}. Ready. 软件急停不能替代物理急停。")
        self.motor_status_var = tk.StringVar(value="Motor: disconnected")
        self.camera_status_var = tk.StringVar(value="Camera: not opened")
        self.focus_var = tk.StringVar(value="Focus index: 0.00")
        self.brightness_var = tk.StringVar(value="Brightness mean: --")
        self.saturation_var = tk.StringVar(value="Saturation fraction: --")
        self.underexposed_var = tk.StringVar(value="Underexposed fraction: --")
        self.exposure_warning_var = tk.StringVar(value="Exposure warning: --")
        self.auto_exposure_var = tk.BooleanVar(value=True)
        self.auto_wb_var = tk.BooleanVar(value=True)
        self.exposure_var = tk.StringVar(value="")
        self.gain_var = tk.StringVar(value="")
        self.brightness_setting_var = tk.StringVar(value="")
        self.contrast_var = tk.StringVar(value="")
        self.wb_temperature_var = tk.StringVar(value="")
        self.camera_properties_var = tk.StringVar(value="Camera properties: --")
        self.current_z_abs_var = tk.StringVar(value="Current Z absolute: waiting")
        self.current_z_rel_var = tk.StringVar(value="Current Z relative: waiting")
        self.best_focus_var = tk.StringVar(value="Best focus index: --")
        self.best_z_abs_var = tk.StringVar(value="Best Z absolute: --")
        self.best_z_rel_var = tk.StringVar(value="Best Z relative: --")
        self.best_focus_time_var = tk.StringVar(value="Best focus time: --")
        self.focus_recording_var = tk.StringVar(value="Recording: no")
        self.focus_assist_message_var = tk.StringVar(value="Open the camera, then start manual focus assist.")
        self.af_scan_range_var = tk.StringVar(value="20")
        self.af_scan_step_var = tk.StringVar(value="5")
        self.af_speed_var = tk.StringVar(value="2")
        self.af_settle_seconds_var = tk.StringVar(value="0.5")
        self.af_sample_seconds_var = tk.StringVar(value="1.5")
        self.af_near_best_ratio_var = tk.StringVar(value="0.96")
        self.af_status_var = tk.StringVar(value="AF status: idle")
        self.af_offset_var = tk.StringVar(value="Current offset: --")
        self.af_score_var = tk.StringVar(value="Current sample focus index: --")
        self.af_best_var = tk.StringVar(value="Best score: --")
        self.af_best_offset_var = tk.StringVar(value="Best offset: --")
        self.af_final_offset_var = tk.StringVar(value="Final offset: --")
        self.af_confirm_score_var = tk.StringVar(value="Confirm score: --")
        self.af_sample_count_var = tk.StringVar(value="Sample points: 0")
        self.abs_pos_var = tk.StringVar(value="Abs X=0  Y=0  Z=0")
        self.rel_pos_var = tk.StringVar(value="Rel X=0  Y=0  Z=0")
        self.recent_command_var = tk.StringVar(value="Recent command: --")
        self.recent_feedback_var = tk.StringVar(value="Recent feedback: --")
        self.recent_error_var = tk.StringVar(value="Recent error: --")
        self.running_state_var = tk.StringVar(value="Running state: idle")
        self.debug_log_var = tk.StringVar(value=f"debug.log: {Path(__file__).with_name('debug.log')}")

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
        ttk.Button(conn, text="Connect Stage", command=self._connect_motor).grid(row=0, column=2, padx=2)
        ttk.Button(conn, text="Disconnect Stage", command=self._disconnect_motor).grid(row=0, column=3, padx=2)
        ttk.Label(conn, text="Camera").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(conn, textvariable=self.camera_index_var, width=12).grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Combobox(conn, textvariable=self.camera_backend_var, values=["DSHOW", "MSMF", "ANY"], width=8, state="readonly").grid(
            row=1, column=2, sticky="ew", padx=2, pady=(6, 0)
        )
        ttk.Button(conn, text="Open Camera / Switch Camera", command=self._open_camera).grid(row=1, column=3, padx=2, pady=(6, 0))
        ttk.Button(conn, text="Refresh Ports", command=self._refresh_serial_ports).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(conn, text="Close Camera", command=self._close_camera).grid(row=2, column=2, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(conn, textvariable=self.motor_status_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(conn, textvariable=self.camera_status_var).grid(row=4, column=0, columnspan=4, sticky="w")

        modes = ttk.LabelFrame(left, text="Mode Selector", padding=8)
        modes.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        modes.columnconfigure(0, weight=1)
        for row, mode in enumerate(Mode):
            ttk.Radiobutton(
                modes,
                text=mode.value,
                value=mode.value,
                variable=self.mode_var,
                command=self._on_mode_change,
            ).grid(row=row, column=0, sticky="w")
        ttk.Label(modes, textvariable=self.current_mode_var, foreground="#005a8d").grid(row=3, column=0, sticky="w", pady=(6, 0))

        move = ttk.LabelFrame(left, text="Manual Move", padding=8)
        move.grid(row=2, column=0, sticky="ew", pady=(0, 8))
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
        ttk.Button(move, text="Software Emergency Stop  Esc", command=self._emergency_stop).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        ttk.Button(move, text="Set Current As Software Origin", command=self._set_software_origin).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )

        pos = ttk.LabelFrame(left, text="Position", padding=8)
        pos.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(pos, textvariable=self.abs_pos_var).grid(row=0, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.rel_pos_var).grid(row=1, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.recent_command_var).grid(row=2, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.recent_feedback_var).grid(row=3, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.recent_error_var).grid(row=4, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.running_state_var).grid(row=5, column=0, sticky="w")
        ttk.Label(pos, textvariable=self.debug_log_var, wraplength=310).grid(row=6, column=0, sticky="w")

        if self.enable_focus_assist:
            focus = ttk.LabelFrame(left, text="手动辅助对焦 Manual Focus Assist", padding=8)
            focus.grid(row=4, column=0, sticky="ew", pady=(0, 8))
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

        if self.enable_autofocus:
            autofocus = ttk.LabelFrame(left, text="Conservative Full Scan Autofocus / 保守 Z 轴自动对焦", padding=8)
            autofocus.grid(row=5, column=0, sticky="ew", pady=(0, 8))
            for col in range(4):
                autofocus.columnconfigure(col, weight=1)
            ttk.Label(autofocus, text="Range").grid(row=0, column=0, sticky="w")
            ttk.Entry(autofocus, textvariable=self.af_scan_range_var, width=7).grid(row=0, column=1, sticky="ew", padx=2)
            ttk.Label(autofocus, text="Step").grid(row=0, column=2, sticky="w")
            ttk.Entry(autofocus, textvariable=self.af_scan_step_var, width=7).grid(row=0, column=3, sticky="ew", padx=2)
            ttk.Label(autofocus, text="Speed %").grid(row=1, column=0, sticky="w")
            ttk.Entry(autofocus, textvariable=self.af_speed_var, width=7).grid(row=1, column=1, sticky="ew", padx=2)
            ttk.Label(autofocus, text="Settle s").grid(row=1, column=2, sticky="w")
            ttk.Entry(autofocus, textvariable=self.af_settle_seconds_var, width=7).grid(row=1, column=3, sticky="ew", padx=2)
            ttk.Label(autofocus, text="Sample s").grid(row=2, column=0, sticky="w")
            ttk.Entry(autofocus, textvariable=self.af_sample_seconds_var, width=7).grid(row=2, column=1, sticky="ew", padx=2)
            ttk.Label(autofocus, text="Near best").grid(row=2, column=2, sticky="w")
            ttk.Entry(autofocus, textvariable=self.af_near_best_ratio_var, width=7).grid(row=2, column=3, sticky="ew", padx=2)
            ttk.Button(autofocus, text="Start Autofocus", command=self._start_autofocus).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2), padx=(0, 2))
            ttk.Button(autofocus, text="Stop Autofocus", command=self._stop_autofocus).grid(row=3, column=2, columnspan=2, sticky="ew", pady=(8, 2), padx=(2, 0))
            labels = [
                self.af_status_var,
                self.af_offset_var,
                self.af_score_var,
                self.af_best_var,
                self.af_best_offset_var,
                self.af_final_offset_var,
                self.af_confirm_score_var,
                self.af_sample_count_var,
            ]
            for index, variable in enumerate(labels, start=4):
                ttk.Label(autofocus, textvariable=variable).grid(row=index, column=0, columnspan=4, sticky="w")
            self.af_canvas = tk.Canvas(autofocus, width=280, height=120, bg="#101820", highlightthickness=0)
            self.af_canvas.grid(row=12, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        else:
            self.af_canvas = None

        camera_controls = ttk.LabelFrame(left, text="Camera Controls", padding=8)
        camera_controls.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            camera_controls.columnconfigure(col, weight=1)
        ttk.Button(camera_controls, text="Read Camera Properties", command=self._read_camera_properties).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Button(camera_controls, text="Reduce Overexposure", command=self._reduce_overexposure).grid(row=0, column=2, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Checkbutton(camera_controls, text="Auto Exposure", variable=self.auto_exposure_var).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(camera_controls, text="Auto White Balance", variable=self.auto_wb_var).grid(row=1, column=2, columnspan=2, sticky="w")
        settings = [
            ("Exposure", self.exposure_var),
            ("Gain", self.gain_var),
            ("Brightness", self.brightness_setting_var),
            ("Contrast", self.contrast_var),
            ("WB Temp", self.wb_temperature_var),
        ]
        for index, (label, variable) in enumerate(settings, start=2):
            row = 2 + (index - 2) // 2
            col = ((index - 2) % 2) * 2
            ttk.Label(camera_controls, text=label).grid(row=row, column=col, sticky="w")
            ttk.Entry(camera_controls, textvariable=variable, width=8).grid(row=row, column=col + 1, sticky="ew", padx=2)
        ttk.Button(camera_controls, text="Apply Camera Settings", command=self._apply_camera_settings).grid(row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=(6, 2))
        ttk.Button(camera_controls, text="Reset Auto Camera Mode", command=self._reset_auto_camera_mode).grid(row=5, column=2, columnspan=2, sticky="ew", padx=2, pady=(6, 2))
        ttk.Label(camera_controls, textvariable=self.camera_properties_var, wraplength=310).grid(row=6, column=0, columnspan=4, sticky="w")

        focus_info = ttk.LabelFrame(left, text="Focus Info", padding=8)
        focus_info.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        for row, variable in enumerate(
            [self.focus_var, self.brightness_var, self.saturation_var, self.underexposed_var, self.exposure_warning_var]
        ):
            ttk.Label(focus_info, textvariable=variable).grid(row=row, column=0, sticky="w")

        mode_panel = ttk.LabelFrame(left, text="Mode-specific panel", padding=8)
        mode_panel.grid(row=8, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(mode_panel, textvariable=self.mode_message_var, wraplength=310).grid(row=0, column=0, sticky="w")

        warning = ttk.Label(left, textvariable=self.status_var, wraplength=310, foreground="#8a4b00")
        warning.grid(row=9, column=0, sticky="ew")

        self.video_label = ttk.Label(right, text="No camera")
        self.video_label.grid(row=0, column=0, sticky="nsew")

    def _bind_keys(self) -> None:
        self.bind("<KeyPress-a>", lambda event: self._axis_key(event, "X", -1))
        self.bind("<KeyPress-d>", lambda event: self._axis_key(event, "X", +1))
        self.bind("<KeyPress-w>", lambda event: self._axis_key(event, "Y", +1))
        self.bind("<KeyPress-s>", lambda event: self._axis_key(event, "Y", -1))
        self.bind("<KeyPress-q>", lambda event: self._axis_key(event, "Z", -1))
        self.bind("<KeyPress-e>", lambda event: self._axis_key(event, "Z", +1))
        self.bind("<space>", lambda _e: self._stop_all())
        self.bind("<Escape>", lambda _e: self._emergency_stop())

    def _axis_key(self, event, axis: str, direction: int) -> None:
        if should_ignore_axis_shortcut(getattr(event, "widget", None)):
            return
        self._move(axis, direction)

    def _current_mode(self) -> Mode:
        try:
            return Mode(self.mode_var.get())
        except ValueError:
            return Mode.MANUAL

    def _on_mode_change(self) -> None:
        mode = self._current_mode()
        self.current_mode_var.set(f"Current mode: {mode.value}")
        messages = {
            Mode.MANUAL: "Manual Mode: manual X/Y/Z control only; no recording and no automatic movement.",
            Mode.FOCUS_ASSIST: "Manual Focus Assist: manually move Z while the GUI records best focus; Go To Best Z only moves Z after confirmation.",
            Mode.AUTO_FOCUS: "Auto Focus: conservative full-scan autofocus. It only moves Z. Manual movement is disabled while AF runs.",
        }
        if mode != Mode.FOCUS_ASSIST:
            self.manual_focus_assist.recording = False
            self._refresh_focus_assist_labels()
        self.mode_message_var.set(messages[mode])
        LOG.info("mode switched: %s", mode.value)
        self._set_status(f"Mode switched to {mode.value}. 软件急停不能替代物理急停。")

    def _connect_motor(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            self._set_status("Serial port is empty. Click Refresh Ports or type the controller port.")
            return
        if sys.platform.startswith("win") and not selected_serial_port_is_listed(port):
            self._set_status(
                f"{port} is not currently listed by Windows. Click Refresh Ports and select the detected controller port."
            )
            return
        self._set_status(f"Opening motor controller on {port}...")

        def worker() -> None:
            controller = None
            try:
                controller = StageController(port=port, invert_x=False, invert_y=False, invert_z=False)
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

    def _disconnect_motor(self) -> None:
        if not self.controller:
            self._set_status("Motor is already disconnected.")
            return
        try:
            if self.controller.is_open:
                self.controller.stop_all()
            self.controller.close()
            self.motor_status_var.set("Motor: disconnected")
            self._set_status("Motor disconnected.")
        except Exception as exc:
            LOG.exception("motor disconnect failed")
            self._set_status(f"Motor disconnect failed: {exc}")

    def _refresh_serial_ports(self) -> None:
        choices = serial_port_choices()
        self.port_combo.configure(values=choices)
        if self.port_var.get().strip() and self.port_var.get().strip() in choices:
            self._set_status("Serial ports refreshed.")
            return
        if sys.platform.startswith("win"):
            self.port_var.set(default_serial_port())
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
        backend = self.camera_backend_var.get().strip().upper() or "DSHOW"
        ok = self.camera.open(index, backend)
        if ok:
            self.focus_rois = None
            self.camera_status_var.set(f"Camera: opened index {index} backend {backend}")
            self._set_status("Camera opened.")
        else:
            self.camera_status_var.set("Camera: no-camera mode")
            self.video_label.configure(text="No camera", image="")
            self._set_status("Camera could not be opened; GUI remains available.")

    def _close_camera(self) -> None:
        self.camera.close()
        self.focus_rois = None
        self.camera_status_var.set("Camera: closed")
        self.video_label.configure(text="No camera", image="")
        self._set_status("Camera closed.")

    def _read_camera_properties(self) -> None:
        if not self.camera.is_open:
            self._set_status("Camera is not open.")
            return
        props = self.camera.get_camera_properties()
        self.camera_properties_var.set(
            "Camera properties: "
            f"exposure={props.get('exposure')} gain={props.get('gain')} "
            f"brightness={props.get('brightness')} contrast={props.get('contrast')}"
        )
        self._set_status("Camera properties read.")

    def _apply_camera_settings(self) -> None:
        if not self.camera.is_open:
            self._set_status("Camera is not open.")
            return
        settings = {
            "auto_exposure": self.auto_exposure_var.get(),
            "auto_wb": self.auto_wb_var.get(),
            "exposure": self._optional_float(self.exposure_var.get()),
            "gain": self._optional_float(self.gain_var.get()),
            "brightness": self._optional_float(self.brightness_setting_var.get()),
            "contrast": self._optional_float(self.contrast_var.get()),
            "wb_temperature": self._optional_float(self.wb_temperature_var.get()),
        }
        self.camera.apply_camera_settings(settings)
        self._set_status("Camera settings applied.")

    def _reduce_overexposure(self) -> None:
        if not self.camera.is_open:
            self._set_status("Camera is not open.")
            return
        result = self.camera.reduce_overexposure()
        self._set_status(
            "Reduce Overexposure: "
            f"success={result['success']} exposure={result['exposure']} "
            f"gain={result['gain']} saturation={result['saturation_fraction']}"
        )

    def _reset_auto_camera_mode(self) -> None:
        if not self.camera.is_open:
            self._set_status("Camera is not open.")
            return
        self.camera.apply_camera_settings({"auto_exposure": True, "auto_wb": True})
        self.auto_exposure_var.set(True)
        self.auto_wb_var.set(True)
        self._set_status("Camera auto exposure and auto white balance requested.")

    def _optional_float(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            self._set_status(f"Camera setting is not a number: {text}")
            return None

    def _move(self, axis: str, direction: int) -> None:
        if self.autofocus.running:
            self._set_status("Autofocus is running; manual movement is disabled except Space/Esc.")
            return
        if not self.controller or not self.controller.is_open:
            self._set_status("Motor is not connected.")
            return
        try:
            pulses = max(1, int(float(self.step_var.get())))
            speed = max(1, min(100, int(float(self.speed_var.get()))))
        except ValueError:
            self._set_status("Step and speed must be numbers.")
            return
        pulses, speed, clamped = clamp_manual_motion_params(pulses, speed)
        if clamped:
            self.step_var.set(str(pulses))
            self.speed_var.set(str(speed))
            self._set_status("SAFE_MODE enabled: parameters clamped.")
            LOG.warning("SAFE_MODE enabled: manual movement parameters clamped to step=%s speed=%s", pulses, speed)
        LOG.info("manual move requested: axis=%s direction=%s pulses=%s speed=%s", axis, direction, pulses, speed)
        self.recent_command_var.set(f"Recent command: Move {axis}{'+' if direction > 0 else '-'} {pulses} pulses @ {speed}%")
        self.running_state_var.set("Running state: manual move")

        def worker() -> None:
            try:
                controller_direction = logical_direction_to_controller_direction(axis, direction)
                self.controller.move_relative(axis, controller_direction, pulses, speed)
                positions = self._safe_read_positions(self.controller)
                self.device_queue.put(("positions", positions))
            except Exception as exc:
                LOG.exception("manual move failed")
                self.device_queue.put(("error", f"Move {axis}{'+' if direction > 0 else '-'} failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_all(self) -> None:
        self.recent_command_var.set("Recent command: Stop all axes")
        if self.autofocus.running:
            self.autofocus.stop_requested = True
            LOG.info("Stop Autofocus requested by Space/controlled stop")
            self.device_queue.put(("af_status", "AF status: stop requested"))
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
        self.recent_command_var.set("Recent command: Software emergency stop")
        self._set_status("SOFTWARE emergency stop sent. 软件急停不能替代物理急停。")
        if self.autofocus.running:
            self.autofocus.stop_requested = True
            LOG.warning("Emergency Stop during AF")
            self.device_queue.put(("af_status", "AF status: emergency stop requested"))
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
        self.mode_var.set(Mode.FOCUS_ASSIST.value)
        self._on_mode_change()
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
                controller_direction = logical_direction_to_controller_direction("Z", direction)
                self.controller.move_relative("Z", controller_direction, pulses, speed)
                positions = self._safe_read_positions(self.controller)
                self.device_queue.put(("positions", positions))
                LOG.info("Go To Best Z completed")
                self.device_queue.put(("status", "Moved to recorded best Z."))
            except Exception as exc:
                LOG.exception("go to best z failed")
                self.device_queue.put(("error", f"Go To Best Z failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _start_autofocus(self) -> None:
        if not self.enable_autofocus:
            return
        self.mode_var.set(Mode.AUTO_FOCUS.value)
        self._on_mode_change()
        if self.autofocus.running:
            self._set_status("Autofocus is already running.")
            return
        if not self.camera.is_open:
            self._set_status("相机未打开，无法自动对焦")
            return
        if not self.controller or not self.controller.is_open:
            self._set_status("电机未连接，无法自动对焦")
            return
        try:
            params = self._read_autofocus_params()
        except ValueError as exc:
            self._set_status(str(exc))
            return
        params, clamped = clamp_autofocus_params(params)
        if clamped:
            message = (
                "SAFE_MODE enabled: parameters clamped to "
                f"range={params.scan_range}, step={params.scan_step}, speed={params.autofocus_speed}"
            )
            LOG.warning(message)
            self._set_status(message)
            self._set_autofocus_param_vars(params)

        saturation = float(self.last_focus_info.get("frame_saturation", 0.0))
        if saturation > 0.30:
            continue_anyway = messagebox.askyesno(
                "Severe Overexposure Warning",
                "当前画面严重过曝，focus index 可能不可靠。建议先使用 Reduce Overexposure。\n\n是否仍然继续？",
            )
            if not continue_anyway:
                self._set_status("Autofocus cancelled because image is severely overexposed.")
                return

        confirmed = messagebox.askyesno(
            "Confirm Autofocus",
            "即将自动移动 Z 轴进行对焦。\n"
            "请确认样品/探针/镜头安全。\n"
            "请确认物理急停可用。\n"
            "第一次测试建议 scan_range <= 20, scan_step <= 5, speed <= 2。\n"
            "软件急停不能替代物理急停。\n"
            "是否继续？",
        )
        if not confirmed:
            self._set_status("Autofocus cancelled.")
            return

        self.autofocus.reset()
        self.autofocus.running = True
        self.running_state_var.set("Running state: autofocus")
        self.recent_command_var.set("Recent command: Start Autofocus")
        self.af_status_var.set("AF status: running")
        self._refresh_autofocus_labels()
        self._draw_autofocus_plot()
        LOG.info("Start Autofocus")
        LOG.info("Autofocus parameters: %s", params)
        threading.Thread(target=self._autofocus_worker, args=(params,), daemon=True).start()

    def _stop_autofocus(self) -> None:
        if not self.autofocus.running:
            self._set_status("Autofocus is not running.")
            return
        self.autofocus.stop_requested = True
        LOG.info("Stop Autofocus")
        self.af_status_var.set("AF status: stop requested")
        self._set_status("Stopping autofocus...")
        if self.controller and self.controller.is_open:
            threading.Thread(target=self._safe_stop_worker, daemon=True).start()

    def _read_autofocus_params(self) -> AutofocusParams:
        try:
            params = AutofocusParams(
                scan_range=max(1, int(float(self.af_scan_range_var.get()))),
                scan_step=max(1, int(float(self.af_scan_step_var.get()))),
                autofocus_speed=max(1, min(100, int(float(self.af_speed_var.get())))),
                settle_seconds=max(0.0, float(self.af_settle_seconds_var.get())),
                sample_seconds=max(0.1, float(self.af_sample_seconds_var.get())),
                near_best_ratio=max(0.1, min(1.0, float(self.af_near_best_ratio_var.get()))),
            )
        except ValueError as exc:
            raise ValueError("Autofocus parameters must be numbers.") from exc
        return params

    def _set_autofocus_param_vars(self, params: AutofocusParams) -> None:
        self.af_scan_range_var.set(str(params.scan_range))
        self.af_scan_step_var.set(str(params.scan_step))
        self.af_speed_var.set(str(params.autofocus_speed))
        self.af_settle_seconds_var.set(str(params.settle_seconds))
        self.af_sample_seconds_var.set(str(params.sample_seconds))
        self.af_near_best_ratio_var.set(str(params.near_best_ratio))

    def _autofocus_worker(self, params: AutofocusParams) -> None:
        current_offset = 0
        points: list[AutofocusSamplePoint] = []
        try:
            self.device_queue.put(("af_status", "AF status: baseline sampling"))
            self._sleep_with_autofocus_stop(params.settle_seconds)
            baseline_score, baseline_iqr, baseline_count = self.sample_focus_at_current_z(params.sample_seconds)
            baseline = AutofocusSamplePoint(0, baseline_score, baseline_iqr, baseline_count)
            points.append(baseline)
            LOG.info("Autofocus baseline: score=%.3f iqr=%.3f frames=%s", baseline_score, baseline_iqr, baseline_count)
            self.device_queue.put(("af_point", baseline))

            self._raise_if_autofocus_stopped()
            self._move_z_for_autofocus(-params.scan_range, params.autofocus_speed)
            current_offset = -params.scan_range
            offsets = build_scan_offsets(params.scan_range, params.scan_step)

            for offset in offsets:
                self._raise_if_autofocus_stopped()
                delta = offset - current_offset
                if delta:
                    self._move_z_for_autofocus(delta, params.autofocus_speed)
                    current_offset = offset
                self.autofocus.current_offset = current_offset
                self.device_queue.put(("af_offset", current_offset))
                self._sleep_with_autofocus_stop(params.settle_seconds)
                score, iqr, frame_count = self.sample_focus_at_current_z(params.sample_seconds)
                point = AutofocusSamplePoint(offset, score, iqr, frame_count)
                LOG.info(
                    "Autofocus point: offset=%s score=%.3f iqr=%.3f frames=%s",
                    offset,
                    score,
                    iqr,
                    frame_count,
                )
                points.append(point)
                self.device_queue.put(("af_point", point))

            if not points:
                raise RuntimeError("no autofocus samples collected")
            peak_score = max(point.score for point in points)
            final_point = select_final_autofocus_point(points, params.near_best_ratio)
            LOG.info("Autofocus peak_score=%.3f", peak_score)
            LOG.info("Autofocus final_offset=%s", final_point.offset)
            self.autofocus.final_offset = final_point.offset
            self.device_queue.put(("af_final", final_point.offset))

            delta_back = final_point.offset - current_offset
            if delta_back:
                self._move_z_for_autofocus(delta_back, params.autofocus_speed)
                current_offset = final_point.offset
            self._sleep_with_autofocus_stop(params.settle_seconds)
            confirm_score, confirm_iqr, confirm_count = self.sample_focus_at_current_z(params.sample_seconds)
            LOG.info(
                "Autofocus confirm_score=%.3f iqr=%.3f frames=%s",
                confirm_score,
                confirm_iqr,
                confirm_count,
            )
            self.device_queue.put(("af_confirm", (confirm_score, confirm_iqr)))
            if confirm_score >= peak_score * 0.90:
                self.device_queue.put(("af_done", "Autofocus completed."))
            else:
                self.device_queue.put(
                    (
                        "af_done",
                        "自动对焦完成，但确认分数偏低，建议重试或使用手动辅助对焦。",
                    )
                )
            LOG.info("Autofocus completed")
        except InterruptedError:
            LOG.info("Autofocus stopped")
            self.device_queue.put(("af_done", "Autofocus stopped."))
        except Exception as exc:
            LOG.exception("autofocus failed")
            if self.controller and self.controller.is_open:
                try:
                    self.controller.stop_all()
                except Exception:
                    LOG.exception("stop_all after autofocus failure failed")
            self.device_queue.put(("af_error", f"Autofocus failed: {exc}"))

    def _move_z_for_autofocus(self, delta: int, speed: int) -> None:
        if delta == 0:
            return
        self._raise_if_autofocus_stopped()
        direction = 1 if delta > 0 else -1
        controller_direction = logical_direction_to_controller_direction("Z", direction)
        self.controller.move_relative("Z", controller_direction, abs(delta), speed)

    def _sleep_with_autofocus_stop(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._raise_if_autofocus_stopped()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _raise_if_autofocus_stopped(self) -> None:
        if self.autofocus.stop_requested:
            raise InterruptedError("autofocus stopped")

    def sample_focus_at_current_z(self, sample_seconds: float) -> tuple[float, float, int]:
        values: list[float] = []
        failures = 0
        deadline = time.monotonic() + sample_seconds
        while time.monotonic() < deadline:
            self._raise_if_autofocus_stopped()
            try:
                with self.camera_lock:
                    frame = self.camera.read_frame()
                    if self.focus_rois is None:
                        self.focus_rois = auto_select_rois(frame)
                    focus_index = calculate_focus_index(frame, self.focus_rois)
                values.append(float(focus_index))
                failures = 0
            except Exception:
                failures += 1
                LOG.exception("autofocus camera sample failed")
                if failures > 10:
                    raise RuntimeError("camera sampling failed more than 10 consecutive times")
                time.sleep(0.05)
        if len(values) < MIN_SAMPLE_FRAMES:
            raise RuntimeError(f"not enough valid focus frames: {len(values)}")
        score = robust_representative(values)
        iqr = _interquartile_range(values)
        return score, iqr, len(values)

    def _read_speed(self, default: int = 10) -> int:
        try:
            _pulses, speed, _changed = clamp_manual_motion_params(1, int(float(self.speed_var.get())))
            return speed
        except ValueError:
            return default

    def _safe_read_positions(self, controller: StageController) -> dict[str, int]:
        try:
            return controller.read_all_positions()
        except Exception:
            LOG.exception("read_all_positions failed; trying per-axis reads")
            return {axis: controller.read_position(axis) for axis in ("X", "Y", "Z")}

    def _poll_positions(self) -> None:
        if self._closing:
            return
        if self.controller and self.controller.is_open and not self.autofocus.running and not self._position_poll_running:
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
        self._schedule_after(700, self._poll_positions)

    def _camera_loop(self) -> None:
        if self._closing:
            return
        if self.camera.is_open:
            try:
                with self.camera_lock:
                    frame = self.camera.read_frame()
                    if self.focus_rois is None:
                        self.focus_rois = auto_select_rois(frame)
                    focus_index = calculate_focus_index(frame, self.focus_rois)
                    self.last_focus_info = brightness_diagnostics(frame)
                self._update_focus(focus_index)
                self._update_focus_diagnostics(self.last_focus_info)
                self._show_frame(frame)
            except Exception:
                LOG.exception("camera loop error")
                self.camera_status_var.set("Camera: read error, no-camera mode")
                self.camera.close()
        self._schedule_after(60, self._camera_loop)

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

    def _update_focus_diagnostics(self, info: dict[str, float | bool]) -> None:
        mean_brightness = float(info.get("mean_brightness", 0.0))
        saturation = float(info.get("frame_saturation", 0.0))
        underexposed = float(info.get("underexposed_fraction", 0.0))
        self.brightness_var.set(f"Brightness mean: {mean_brightness:.1f}")
        self.saturation_var.set(f"Saturation fraction: {saturation:.3f}")
        self.underexposed_var.set(f"Underexposed fraction: {underexposed:.3f}")
        if saturation > 0.30:
            warning = "Severe overexposure"
        elif saturation > 0.10:
            warning = "Overexposed warning"
        elif mean_brightness > 220.0:
            warning = "Too bright"
        elif mean_brightness < 20.0:
            warning = "Too dark"
        else:
            warning = "OK"
        self.exposure_warning_var.set(f"Exposure warning: {warning}")

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
        if self._closing:
            return
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
                    self.recent_feedback_var.set(f"Recent feedback: positions {payload}")
                    if not self.autofocus.running:
                        self.running_state_var.set("Running state: idle")
                elif kind == "position_poll_done":
                    self._position_poll_running = False
                elif kind == "status":
                    self._set_status(str(payload))
                elif kind == "error":
                    self.recent_error_var.set(f"Recent error: {payload}")
                    self.running_state_var.set("Running state: error")
                    self._set_status(str(payload))
                elif kind == "af_status":
                    self.af_status_var.set(str(payload))
                elif kind == "af_offset":
                    self.autofocus.current_offset = int(payload)
                    self._refresh_autofocus_labels()
                elif kind == "af_point":
                    self._add_autofocus_point(payload)
                elif kind == "af_final":
                    self.autofocus.final_offset = int(payload)
                    self._refresh_autofocus_labels()
                    self._draw_autofocus_plot()
                elif kind == "af_confirm":
                    confirm_score, confirm_iqr = payload
                    self.autofocus.confirm_score = float(confirm_score)
                    self.autofocus.confirm_iqr = float(confirm_iqr)
                    self._refresh_autofocus_labels()
                elif kind == "af_done":
                    self.autofocus.running = False
                    self.autofocus.stop_requested = False
                    self.running_state_var.set("Running state: idle")
                    self.af_status_var.set(f"AF status: {payload}")
                    self._set_status(str(payload))
                elif kind == "af_error":
                    self.autofocus.running = False
                    self.autofocus.stop_requested = False
                    self.running_state_var.set("Running state: error")
                    self.recent_error_var.set(f"Recent error: {payload}")
                    self.af_status_var.set("AF status: failed")
                    self._set_status(str(payload))
        except queue.Empty:
            pass
        self._schedule_after(50, self._drain_device_queue)

    def _add_autofocus_point(self, point: AutofocusSamplePoint) -> None:
        if self.autofocus.sample_points is None:
            self.autofocus.sample_points = []
        self.autofocus.sample_points.append(point)
        self.autofocus.current_offset = point.offset
        self.autofocus.current_score = point.score
        if self.autofocus.best_score is None or point.score > self.autofocus.best_score:
            self.autofocus.best_score = point.score
            self.autofocus.best_offset = point.offset
        self._refresh_autofocus_labels()
        self._draw_autofocus_plot()

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

    def _refresh_autofocus_labels(self) -> None:
        if not self.enable_autofocus:
            return
        state = self.autofocus
        self.af_offset_var.set(f"Current offset: {state.current_offset}")
        self.af_score_var.set(
            "Current sample focus index: --"
            if state.current_score is None
            else f"Current sample focus index: {state.current_score:.2f}"
        )
        self.af_best_var.set(
            "Best score: --" if state.best_score is None else f"Best score: {state.best_score:.2f}"
        )
        self.af_best_offset_var.set(
            "Best offset: --" if state.best_offset is None else f"Best offset: {state.best_offset}"
        )
        self.af_final_offset_var.set(
            "Final offset: --" if state.final_offset is None else f"Final offset: {state.final_offset}"
        )
        self.af_confirm_score_var.set(
            "Confirm score: --"
            if state.confirm_score is None
            else f"Confirm score: {state.confirm_score:.2f}"
        )
        count = len(state.sample_points or [])
        self.af_sample_count_var.set(f"Sample points: {count}")

    def _draw_autofocus_plot(self) -> None:
        canvas = self.af_canvas
        if canvas is None:
            return
        canvas.delete("all")
        width = max(10, canvas.winfo_width())
        height = max(10, canvas.winfo_height())
        canvas.create_rectangle(0, 0, width, height, fill="#101820", outline="")
        points = self.autofocus.sample_points or []
        if not points:
            return
        offsets = [point.offset for point in points]
        scores = [point.score for point in points]
        x_min = min(offsets)
        x_max = max(offsets)
        y_min = min(scores)
        y_max = max(scores)
        x_span = max(1, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        best_offset = self.autofocus.best_offset
        final_offset = self.autofocus.final_offset
        for point in points:
            x = 8 + ((point.offset - x_min) / x_span) * (width - 16)
            y = height - 8 - ((point.score - y_min) / y_span) * (height - 16)
            radius = 3
            fill = "#4cc9f0"
            if point.offset == best_offset:
                radius = 5
                fill = "#ffd166"
            if final_offset is not None and point.offset == final_offset:
                radius = 6
                fill = "#80ed99"
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="")
        canvas.create_text(8, 8, anchor="nw", fill="#e6edf3", text="offset vs focus")

    def _set_status(self, message: str) -> None:
        LOG.info(message)
        self.status_var.set(message)

    def _on_close(self) -> None:
        self._closing = True
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
