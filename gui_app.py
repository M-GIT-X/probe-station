"""Tkinter GUI for the three-axis automated probe station."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import math
from pathlib import Path
import sys
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from camera_opencv import OpenCVCamera, choose_best_exposure_result, exposure_tuning_candidates
from focus_metrics import (
    auto_select_rois,
    brightness_diagnostics,
    calculate_focus_index,
    robust_representative,
    stabilize_frame_translation,
)
from image_stitcher import IncrementalMosaicBuilder, stitch_session_by_metadata
from sample_plane import (
    SamplePlanePoint,
    boundary_polygon_from_points,
    bounds_from_plane_points,
    evaluate_plane_consistency,
    fit_sample_plane,
    fit_sample_plane_robust,
    generate_plane_probe_points,
)
from scan_plan import TilePoint, generate_overlap_scan_plan
from stage_controller import StageController
from stitching_calibration import StitchingCalibration, build_in_bounds_trial_plan, estimate_calibration_from_frames
from stitching_store import StitchingSessionStore, TileRecord
from stitching_view_model import build_stitching_view_model


LOG = logging.getLogger(__name__)

APP_TITLE = "Three-Axis Probe Station"


class Mode(Enum):
    MANUAL = "Manual Mode"
    AUTO_FOCUS = "Auto Focus"
    IMAGE_STITCHING = "Image Stitching"


class AutofocusMode(Enum):
    SEMI = "Semi Auto"
    FULL = "Full Auto"


class StitchingFocusMode(Enum):
    SEMI = "Semi Auto"
    FULL = "Full Auto"


INVERT_X_DIRECTION = False
INVERT_Y_DIRECTION = True
INVERT_Z_DIRECTION = False

SAFE_MODE = True
SAFE_MAX_MANUAL_STEP = 2000
SAFE_MAX_MANUAL_SPEED = 100
SAFE_MAX_AUTOFOCUS_RANGE = 5000
SAFE_MAX_AUTOFOCUS_STEP = 2000
SAFE_MAX_AUTOFOCUS_SPEED = 100
MIN_SAMPLE_FRAMES = 3
MIN_STITCHING_CORNER_AUTOFOCUS_SAMPLE_SECONDS = 1.0
STITCHING_PLANE_CONFIRMATION_RESIDUAL_PULSES = 30.0
DEFAULT_WINDOW_GEOMETRY = "1400x900"
DEFAULT_CAMERA_INDEX = "1"
MIN_WINDOW_WIDTH = 960
MIN_WINDOW_HEIGHT = 620
VIDEO_PREVIEW_MAX_WIDTH = 1013
VIDEO_PREVIEW_MAX_HEIGHT = 633
LEFT_PANEL_MIN_WIDTH = 320
RIGHT_PANEL_MIN_WIDTH = 360
CENTER_PANEL_MIN_WIDTH = VIDEO_PREVIEW_MAX_WIDTH + 12
BODY_HORIZONTAL_PADDING = 20
STITCHING_PLOT_SIZE = 370
STITCH_PREVIEW_SAMPLE_TARGET = 800
LARGE_STITCH_REDRAW_TILE_INTERVAL = 10
LARGE_STITCH_REDRAW_SECONDS = 0.5
AUTOFOCUS_CURVE_WIDTH = 580
AUTOFOCUS_CURVE_HEIGHT = 130
PREVIEW_PLACEHOLDER_TEXT = "CAMERA OFFLINE\nAwaiting optical feed"
PREVIEW_BACKGROUND_COLOR = "#000000"
CROSSHAIR_COLOR_RGB = (57, 255, 20)
MISSION_LOG_BACKGROUND_COLOR = "#000000"
MISSION_LOG_FOREGROUND_COLOR = "#39ff14"
EMERGENCY_STOP_BACKGROUND_COLOR = "#d45d67"
EMERGENCY_STOP_FOREGROUND_COLOR = "#ffffff"


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


def should_ignore_axis_shortcut(widget) -> bool:
    if widget is None:
        return False
    try:
        widget_class = widget.winfo_class()
    except Exception:
        return False
    return widget_class in {"Entry", "TEntry", "Combobox", "TCombobox", "Spinbox", "TSpinbox"}


def should_release_numeric_entry_focus(char: str, keysym: str) -> bool:
    if keysym in {"BackSpace", "Delete", "Left", "Right", "Home", "End", "Tab", "Return", "KP_Enter"}:
        return False
    if char == "":
        return False
    return not (char.isdigit() or char in {".", "-"})


def should_return_focus_to_root_on_enter(widget) -> bool:
    return should_ignore_axis_shortcut(widget)


def mission_log_message_for_event(kind: str, payload: object = None) -> str | None:
    text = "" if payload is None else str(payload)
    messages = {
        "motor_connecting": f"STAGE LINK INITIATED. Opening controller channel on {text}.",
        "motor_connected": "STAGE ONLINE. Motion controller handshake confirmed.",
        "motor_disconnected": "STAGE OFFLINE. Motion channel secured.",
        "camera_opened": f"CAMERA ONLINE. Optical feed established: {text}.",
        "camera_closed": "CAMERA OFFLINE. Optical feed secured.",
        "camera_exposure_tuning_started": "EXPOSURE CALIBRATION START. Optical feed entering focus readiness sequence.",
        "camera_exposure_tuning_completed": f"EXPOSURE CALIBRATION COMPLETE. {text}.",
        "camera_exposure_tuning_failed": f"EXPOSURE CALIBRATION ABORT. Fault received: {text}.",
        "autofocus_started": "AUTOFOCUS SEQUENCE START. Z-axis scan authorized.",
        "autofocus_completed": "AUTOFOCUS COMPLETE. Final focus position confirmed.",
        "autofocus_stopped": "AUTOFOCUS HOLD. Stop command acknowledged.",
        "autofocus_failed": f"AUTOFOCUS ABORT. Fault received: {text}.",
        "stitch_scan_completed": "SCAN ACQUISITION COMPLETE. All image tiles secured.",
        "stitch_assembling": "MOSAIC ASSEMBLY IN PROGRESS. Processing captured tiles.",
        "stitch_completed": f"MOSAIC COMPLETE. Composite image saved: {text}.",
        "stitch_focus_mode": f"STITCHING FOCUS MODE SET. {text}.",
        "stitch_corner_recorded": f"STITCHING CORNER RECORDED. {text}.",
        "stitch_corner_deleted": f"STITCHING CORNER DELETED. {text}.",
        "stitch_corners_cleared": "STITCHING CORNERS CLEARED. Awaiting four focused boundary points.",
        "stitch_preflight": f"STITCHING PREFLIGHT REPORT. {text}.",
        "stitch_started": f"IMAGE STITCHING SEQUENCE START. {text}.",
        "stitch_cancelled": "IMAGE STITCHING SEQUENCE HOLD. Operator cancelled before motion.",
        "stitch_stop_requested": "IMAGE STITCHING SEQUENCE HOLD. Stop command acknowledged.",
        "stitch_failed": f"IMAGE STITCHING ABORT. Fault received: {text}.",
        "stitch_semiauto_plane": "SEMI AUTO FOCUS MODE. Using operator-focused corner Z positions.",
        "stitch_fullauto_plane": "FULL AUTO FOCUS MODE. Probing a 3x3 autofocus grid for robust plane fitting.",
        "controlled_stop": "CONTROLLED STOP COMMAND SENT. All axes ordered to halt.",
        "emergency_stop": "SOFTWARE EMERGENCY STOP SENT. Verify physical system state immediately.",
    }
    return messages.get(kind)


def manual_shortcut_mapping(key: str) -> tuple[str, int] | None:
    mapping = {
        "a": ("X", +1),
        "d": ("X", -1),
        "w": ("Y", -1),
        "s": ("Y", +1),
        "q": ("Z", -1),
        "e": ("Z", +1),
    }
    return mapping.get(key.lower())


def draw_preview_crosshair(
    rgb_frame: np.ndarray,
    *,
    cursor_position: tuple[int, int] | None = None,
    include_center: bool = True,
    cursor_radius: int = 8,
) -> np.ndarray:
    overlaid = rgb_frame.copy()
    if include_center:
        center_y = overlaid.shape[0] // 2
        center_x = overlaid.shape[1] // 2
        overlaid[center_y, :, :3] = CROSSHAIR_COLOR_RGB
        overlaid[:, center_x, :3] = CROSSHAIR_COLOR_RGB
    if cursor_position is not None:
        cursor_x, cursor_y = cursor_position
        if 0 <= cursor_x < overlaid.shape[1] and 0 <= cursor_y < overlaid.shape[0]:
            radius = max(1, int(cursor_radius))
            left = max(0, cursor_x - radius)
            right = min(overlaid.shape[1], cursor_x + radius + 1)
            top = max(0, cursor_y - radius)
            bottom = min(overlaid.shape[0], cursor_y + radius + 1)
            overlaid[cursor_y, left:right, :3] = CROSSHAIR_COLOR_RGB
            overlaid[top:bottom, cursor_x, :3] = CROSSHAIR_COLOR_RGB
    return overlaid


def stage_delta_from_preview_delta(
    dx_pixels: float,
    dy_pixels: float,
    calibration: StitchingCalibration,
) -> tuple[int, int]:
    if calibration.x_pixels_per_pulse == 0.0 or calibration.y_pixels_per_pulse == 0.0:
        raise ValueError("pixel-per-pulse calibration must be non-zero")
    return (
        int(round(float(dx_pixels) / calibration.x_pixels_per_pulse)),
        int(round(float(dy_pixels) / calibration.y_pixels_per_pulse)),
    )


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


def absolute_delta_to_controller_direction(axis: str, *, current: int, target: int) -> int:
    """Convert a native controller-position delta directly to its move sign."""
    del axis
    delta = int(target) - int(current)
    return 1 if delta > 0 else -1 if delta < 0 else 0


def clamp_manual_motion_params(pulses: int, speed: int) -> tuple[int, int, bool]:
    pulses = max(1, int(pulses))
    speed = max(1, min(100, int(speed)))
    if not SAFE_MODE:
        return pulses, speed, False
    clamped_pulses = min(pulses, SAFE_MAX_MANUAL_STEP)
    clamped_speed = min(speed, SAFE_MAX_MANUAL_SPEED)
    return clamped_pulses, clamped_speed, (clamped_pulses, clamped_speed) != (pulses, speed)


@dataclass(frozen=True)
class ModePanelSpec:
    visible_sections: tuple[str, ...]
    primary_actions: tuple[str, ...]
    status_fields: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class ScrollableBodyLayoutSpec:
    vertical_scrollbar: bool
    horizontal_scrollbar: bool
    min_window_size: tuple[int, int]
    content_min_width: int


def scrollable_body_layout_spec() -> ScrollableBodyLayoutSpec:
    return ScrollableBodyLayoutSpec(
        vertical_scrollbar=True,
        horizontal_scrollbar=True,
        min_window_size=(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT),
        content_min_width=LEFT_PANEL_MIN_WIDTH + CENTER_PANEL_MIN_WIDTH + RIGHT_PANEL_MIN_WIDTH + BODY_HORIZONTAL_PADDING,
    )


def mode_panel_spec(mode: Mode | str) -> ModePanelSpec:
    selected = mode if isinstance(mode, Mode) else Mode(str(mode))
    if selected == Mode.AUTO_FOCUS:
        return ModePanelSpec(
            visible_sections=("autofocus",),
            primary_actions=("Start Autofocus", "Stop Autofocus"),
            status_fields=("AF status", "Current offset", "Best score", "Final offset", "Focus curve"),
            message="Auto Focus: Start Autofocus runs a conservative Z-only scan; manual movement returns when AF is stopped or complete.",
        )
    if selected == Mode.IMAGE_STITCHING:
        return ModePanelSpec(
            visible_sections=("image_stitching",),
            primary_actions=("Record Corner", "Delete Last Corner", "Start Stitching Scan"),
            status_fields=("Corner count", "Sample plane fit", "Tile progress", "Stitched mosaic"),
            message="Image Stitching: record four XY corners; scan start refocuses corners and fits the Z plane.",
        )
    return ModePanelSpec(
        visible_sections=("manual",),
        primary_actions=("Manual X/Y/Z move", "Stop Space", "Software Emergency Stop Esc"),
        status_fields=("Absolute position", "Relative position", "Live focus index"),
        message="Manual Mode: manual X/Y/Z control only; no recording and no automatic movement.",
    )


def stitching_geometry_input_fields() -> tuple[str, ...]:
    return ("Overlap %",)


def stitching_focus_mode_choices() -> tuple[str, ...]:
    return (StitchingFocusMode.SEMI.value, StitchingFocusMode.FULL.value)


def format_stitching_preflight_report(
    *,
    corner_count: int,
    has_calibration: bool,
    last_focus_info: dict[str, float | bool],
    output_root: str,
    overlap_percent: float,
    sample_frames: int,
) -> str:
    mean_brightness = float(last_focus_info.get("mean_brightness", 0.0))
    saturation = float(last_focus_info.get("frame_saturation", 0.0))
    exposure = "exposure OK"
    if mean_brightness > 225.0 or saturation > 0.10:
        exposure = "bright/overexposed"
    elif mean_brightness < 25.0:
        exposure = "dark"
    calibration = "calibration ready" if has_calibration else "calibration pending"
    return (
        f"Preflight: corners {corner_count}/4, {calibration}, {exposure}, "
        f"overlap {overlap_percent:.1f}%, frames/tile {int(sample_frames)}, output {output_root}"
    )


def preview_tile_draw_indices(
    total_tiles: int,
    *,
    current_tile_index: int | None = None,
    captured_tile_count: int = 0,
    sample_target: int = STITCH_PREVIEW_SAMPLE_TARGET,
) -> set[int]:
    total_tiles = max(0, int(total_tiles))
    if total_tiles <= sample_target:
        return set(range(total_tiles))
    stride = max(1, math.ceil(total_tiles / sample_target))
    indices = set(range(0, total_tiles, stride))
    if current_tile_index is not None and 0 <= current_tile_index < total_tiles:
        indices.add(current_tile_index)
    if captured_tile_count > 0:
        indices.add(min(total_tiles - 1, captured_tile_count - 1))
    indices.add(0)
    indices.add(total_tiles - 1)
    return indices


@dataclass(frozen=True)
class AutofocusParams:
    scan_range: int = 20
    scan_step: int = 5
    autofocus_speed: int = 90
    settle_seconds: float = 0.5
    sample_seconds: float = 1.5
    near_best_ratio: float = 0.96


@dataclass(frozen=True)
class AutofocusPassPlan:
    name: str
    scan_range: int
    scan_step: int


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


@dataclass
class StitchingRunState:
    running: bool = False
    stop_requested: bool = False
    corners: list[SamplePlanePoint] | None = None
    planned_tiles: list[TilePoint] | None = None
    current_tile_index: int | None = None
    captured_tile_count: int = 0
    calibration: StitchingCalibration | None = None
    last_session_path: Path | None = None
    last_mosaic_path: Path | None = None

    def reset_run(self) -> None:
        self.running = False
        self.stop_requested = False
        self.current_tile_index = None
        self.captured_tile_count = 0


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


def build_autofocus_pass_plan(params: AutofocusParams, mode: AutofocusMode | str) -> list[AutofocusPassPlan]:
    selected = mode if isinstance(mode, AutofocusMode) else AutofocusMode(str(mode))
    scan_range = max(1, int(params.scan_range))
    if selected == AutofocusMode.SEMI:
        return [AutofocusPassPlan("semi", scan_range, max(1, int(params.scan_step)))]

    coarse_step = max(5, scan_range // 4)
    mid_range = max(2, scan_range // 2)
    mid_step = max(2, coarse_step // 2)
    fine_range = max(1, scan_range // 5)
    fine_step = 1
    return [
        AutofocusPassPlan("coarse", scan_range, coarse_step),
        AutofocusPassPlan("refine", mid_range, mid_step),
        AutofocusPassPlan("fine", fine_range, fine_step),
    ]


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


class ProbeStationApp(tk.Tk):
    def __init__(self, enable_autofocus: bool = True) -> None:
        super().__init__()
        self.app_stage_title = APP_TITLE
        self.title(APP_TITLE)
        self.geometry(DEFAULT_WINDOW_GEOMETRY)
        self.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self.enable_autofocus = enable_autofocus
        self.controller: StageController | None = None
        self.camera = OpenCVCamera()
        self.camera_lock = threading.Lock()
        self.device_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.absolute_pos = {"X": 0, "Y": 0, "Z": 0}
        self.position_available = {"X": False, "Y": False, "Z": False}
        self.software_origin = {"X": 0, "Y": 0, "Z": 0}
        self.focus_rois = None
        self.focus_reference_frame = None
        self.last_focus_info = {
            "mean_brightness": 0.0,
            "frame_saturation": 0.0,
            "underexposed_fraction": 0.0,
            "overexposed": False,
        }
        self.autofocus = AutofocusRunState()
        self.autofocus.reset()
        self.stitching = StitchingRunState(corners=[])
        self.stage_image_calibration: StitchingCalibration | None = None
        self.preview_cursor_position: tuple[int, int] | None = None
        self.preview_display_info: tuple[int, int, int, int] | None = None
        self._preview_base_rgb: np.ndarray | None = None
        self._inline_autofocus_status_prefix = "Progress: autofocus"
        self._photo = None
        self._placeholder_photo = None
        self._last_frame_time = 0.0
        self._last_stitch_plot_time = 0.0
        self._last_stitch_plot_progress = -1
        self._position_poll_running = False
        self._closing = False
        self._after_ids: set[str] = set()
        self.mission_log_lines: list[str] = []

        self._build_ui()
        self._bind_keys()
        self.bind_all("<Return>", self._return_focus_to_root)
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
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.port_var = tk.StringVar(value=default_serial_port())
        self.camera_index_var = tk.StringVar(value=DEFAULT_CAMERA_INDEX)
        self.camera_backend_var = tk.StringVar(value="DSHOW")
        self.mode_var = tk.StringVar(value=Mode.MANUAL.value)
        self.current_mode_var = tk.StringVar(value=f"Current mode: {Mode.MANUAL.value}")
        self.step_var = tk.StringVar(value="10")
        self.speed_var = tk.StringVar(value="90")
        self.motion_control_mode_var = tk.StringVar(value="Keyboard")
        self.stage_calibration_var = tk.StringVar(value="XY calibration: not calibrated")
        self.status_var = tk.StringVar(value=f"{APP_TITLE}. Ready. 软件急停不能替代物理急停。")
        self.motor_status_var = tk.StringVar(value="Motor: disconnected")
        self.camera_status_var = tk.StringVar(value="Camera: not opened")
        self.focus_var = tk.StringVar(value="Sharpness index: 0.00")
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
        self.af_scan_range_var = tk.StringVar(value="20")
        self.af_scan_step_var = tk.StringVar(value="5")
        self.af_speed_var = tk.StringVar(value="90")
        self.af_mode_var = tk.StringVar(value=AutofocusMode.SEMI.value)
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
        self.stitch_overlap_var = tk.StringVar(value="25")
        self.stitch_speed_var = tk.StringVar(value="90")
        self.stitch_settle_seconds_var = tk.StringVar(value="0.5")
        self.stitch_sample_frames_var = tk.StringVar(value="5")
        self.stitch_focus_mode_var = tk.StringVar(value=StitchingFocusMode.FULL.value)
        self.stitch_output_root_var = tk.StringVar(value=str(Path(__file__).with_name("stitching_output")))
        self.stitch_corner_var = tk.StringVar(value="Corners: 0/4")
        self.stitch_plane_var = tk.StringVar(value="Plane: not fitted")
        self.stitch_calibration_var = tk.StringVar(value="Calibration: pending")
        self.stitch_plan_var = tk.StringVar(value="Plan: pending")
        self.stitch_progress_var = tk.StringVar(value="Progress: idle")
        self.stitch_preflight_var = tk.StringVar(value="Preflight: idle")
        self.stitch_output_var = tk.StringVar(value="Output: --")
        self.abs_pos_var = tk.StringVar(value="Abs X=0  Y=0  Z=0")
        self.rel_pos_var = tk.StringVar(value="Rel X=0  Y=0  Z=0")
        self.recent_command_var = tk.StringVar(value="Recent command: --")
        self.recent_feedback_var = tk.StringVar(value="Recent feedback: --")
        self.recent_error_var = tk.StringVar(value="Recent error: --")
        self.running_state_var = tk.StringVar(value="Running state: idle")

        top = ttk.Frame(self, padding=(10, 10, 10, 6))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(3, weight=1)
        ttk.Label(top, text=APP_TITLE, font=("", 14, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 24))
        ttk.Label(top, text="Mode").grid(row=0, column=1, sticky="w")
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var, values=[mode.value for mode in Mode], state="readonly", width=22)
        self.mode_combo.grid(row=0, column=2, sticky="w", padx=(6, 18))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_mode_change())
        ttk.Label(top, textvariable=self.current_mode_var, foreground="#005a8d").grid(row=0, column=3, sticky="w")
        ttk.Label(top, textvariable=self.running_state_var).grid(row=0, column=4, sticky="e")

        body_viewport = ttk.Frame(self, padding=(10, 0, 10, 10))
        body_viewport.grid(row=1, column=0, sticky="nsew")
        body_viewport.rowconfigure(0, weight=1)
        body_viewport.columnconfigure(0, weight=1)
        self.body_canvas = tk.Canvas(body_viewport, highlightthickness=0, borderwidth=0)
        self.body_v_scrollbar = ttk.Scrollbar(body_viewport, orient="vertical", command=self.body_canvas.yview)
        self.body_h_scrollbar = ttk.Scrollbar(body_viewport, orient="horizontal", command=self.body_canvas.xview)
        self.body_canvas.configure(
            yscrollcommand=self.body_v_scrollbar.set,
            xscrollcommand=self.body_h_scrollbar.set,
        )
        self.body_canvas.grid(row=0, column=0, sticky="nsew")
        self.body_v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.body_h_scrollbar.grid(row=1, column=0, sticky="ew")

        body = ttk.Frame(self.body_canvas)
        self.body_frame = body
        self._body_window_id = self.body_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", self._update_body_scrollregion)
        self.body_canvas.bind("<Configure>", self._resize_body_canvas_window)
        self.body_canvas.bind("<Enter>", self._bind_body_mousewheel)
        self.body_canvas.bind("<Leave>", self._unbind_body_mousewheel)
        body.columnconfigure(0, minsize=LEFT_PANEL_MIN_WIDTH)
        body.columnconfigure(1, weight=1, minsize=CENTER_PANEL_MIN_WIDTH)
        body.columnconfigure(2, minsize=RIGHT_PANEL_MIN_WIDTH)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        left.grid(row=0, column=0, sticky="ns")
        center = ttk.Frame(body)
        center.grid(row=0, column=1, sticky="nsew")
        center.columnconfigure(0, weight=1)
        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        right.grid(row=0, column=2, sticky="ns")
        right.columnconfigure(0, weight=1)

        conn = ttk.LabelFrame(left, text="Devices", padding=8)
        conn.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        conn.columnconfigure(1, weight=1)
        ttk.Label(conn, text="Serial").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, values=serial_port_choices(), width=11)
        self.port_combo.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(conn, text="Connect", command=self._connect_motor).grid(row=0, column=2, padx=2)
        ttk.Button(conn, text="Disconnect", command=self._disconnect_motor).grid(row=1, column=2, padx=2, pady=(5, 0))
        ttk.Button(conn, text="Refresh Ports", command=self._refresh_serial_ports).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(conn, text="Camera").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(conn, textvariable=self.camera_index_var, width=6).grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        ttk.Combobox(conn, textvariable=self.camera_backend_var, values=["DSHOW", "MSMF", "ANY"], width=7, state="readonly").grid(
            row=3, column=0, sticky="ew", pady=(5, 0)
        )
        ttk.Button(conn, text="Open Camera", command=self._open_camera).grid(row=3, column=1, sticky="ew", padx=4, pady=(5, 0))
        ttk.Button(conn, text="Close", command=self._close_camera).grid(row=3, column=2, sticky="ew", pady=(5, 0))
        ttk.Button(conn, text="Calibrate XY Scale", command=self._start_stage_image_calibration).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Label(conn, textvariable=self.stage_calibration_var, wraplength=270).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(conn, textvariable=self.motor_status_var, wraplength=270).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(conn, textvariable=self.camera_status_var, wraplength=270).grid(row=7, column=0, columnspan=3, sticky="w")

        self.manual_panel = ttk.LabelFrame(left, text="Motion Control", padding=8)
        move = self.manual_panel
        move.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for col in range(3):
            move.columnconfigure(col, weight=1)
        ttk.Label(move, text="Step pulses").grid(row=0, column=0, sticky="w")
        self.step_entry = ttk.Entry(move, textvariable=self.step_var, width=8)
        self.step_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(move, text="Speed %").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.speed_entry = ttk.Entry(move, textvariable=self.speed_var, width=8)
        self.speed_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        for entry in (self.step_entry, self.speed_entry):
            entry.bind("<KeyPress>", self._numeric_entry_keypress)
        ttk.Radiobutton(move, text="Keyboard", value="Keyboard", variable=self.motion_control_mode_var).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Radiobutton(move, text="Mouse", value="Mouse", variable=self.motion_control_mode_var).grid(
            row=1, column=2, sticky="w", pady=(4, 0)
        )
        self.manual_move_buttons = [
            ttk.Button(move, text="Y-  W", command=lambda: self._move("Y", -1)),
            ttk.Button(move, text="X+  A", command=lambda: self._move("X", +1)),
            ttk.Button(move, text="X-  D", command=lambda: self._move("X", -1)),
            ttk.Button(move, text="Y+  S", command=lambda: self._move("Y", +1)),
            ttk.Button(move, text="Z-  Q", command=lambda: self._move("Z", -1)),
            ttk.Button(move, text="Z+  E", command=lambda: self._move("Z", +1)),
        ]
        self.manual_move_buttons[0].grid(row=2, column=1, sticky="ew", pady=(10, 2))
        self.manual_move_buttons[1].grid(row=3, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(move, text="Stop Space", command=self._stop_all).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        self.manual_move_buttons[2].grid(row=3, column=2, sticky="ew", padx=2, pady=2)
        self.manual_move_buttons[3].grid(row=4, column=1, sticky="ew", pady=2)
        self.manual_move_buttons[4].grid(row=5, column=0, sticky="ew", padx=2, pady=(10, 2))
        self.manual_move_buttons[5].grid(row=5, column=2, sticky="ew", padx=2, pady=(10, 2))
        self.emergency_stop_button = tk.Button(
            move,
            text="Software Emergency Stop  Esc",
            command=self._emergency_stop,
            bg=EMERGENCY_STOP_BACKGROUND_COLOR,
            fg=EMERGENCY_STOP_FOREGROUND_COLOR,
            activebackground="#bb4550",
            activeforeground=EMERGENCY_STOP_FOREGROUND_COLOR,
            relief="raised",
        )
        self.emergency_stop_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.software_origin_button = ttk.Button(move, text="Set Software Origin", command=self._set_software_origin)
        self.software_origin_button.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        self.focus_info_panel = ttk.LabelFrame(left, text="Sharpness / Image Status", padding=8)
        focus_info = self.focus_info_panel
        focus_info.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for variable in [self.focus_var, self.brightness_var, self.saturation_var, self.underexposed_var, self.exposure_warning_var]:
            ttk.Label(focus_info, textvariable=variable).pack(anchor="w")

        self.position_activity_panel = ttk.LabelFrame(left, text="Position / Activity", padding=8)
        pos = self.position_activity_panel
        pos.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        for variable in [self.abs_pos_var, self.rel_pos_var, self.recent_command_var, self.recent_feedback_var, self.recent_error_var]:
            ttk.Label(pos, textvariable=variable, wraplength=300).pack(anchor="w")

        video_frame = ttk.LabelFrame(center, text="Camera Preview", padding=4)
        video_frame.grid(row=0, column=0, sticky="n", pady=(0, 8))
        self.video_label = tk.Label(
            video_frame,
            text=PREVIEW_PLACEHOLDER_TEXT,
            anchor="center",
            bg=PREVIEW_BACKGROUND_COLOR,
            fg="#9aa5b1",
            compound="center",
            borderwidth=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Motion>", self._on_preview_mouse_motion)
        self.video_label.bind("<Leave>", self._on_preview_mouse_leave)
        self.video_label.bind("<Double-Button-1>", self._on_preview_double_click)
        self._show_camera_placeholder()

        self.af_plot_frame = ttk.LabelFrame(center, text="Autofocus Curve", padding=4)
        self.af_plot_frame.grid(row=1, column=0, sticky="n", pady=(0, 8))
        self.af_canvas = tk.Canvas(
            self.af_plot_frame,
            width=AUTOFOCUS_CURVE_WIDTH,
            height=AUTOFOCUS_CURVE_HEIGHT,
            bg="#101820",
            highlightthickness=0,
        )
        self.af_canvas.grid(row=0, column=0)

        self.stitch_plot_frame = ttk.LabelFrame(center, text="Stitching Plane / Progress", padding=4)
        self.stitch_plot_frame.grid(row=1, column=0, sticky="n", pady=(0, 8))
        self.stitch_plane_canvas = tk.Canvas(
            self.stitch_plot_frame,
            width=STITCHING_PLOT_SIZE,
            height=STITCHING_PLOT_SIZE,
            bg="#101820",
            highlightthickness=0,
        )
        self.stitch_plane_canvas.grid(row=0, column=0)
        self._draw_stitching_plane_view()

        if self.enable_autofocus:
            self.autofocus_panel = ttk.LabelFrame(right, text="Auto Focus Controls", padding=8)
            autofocus = self.autofocus_panel
            autofocus.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            for col in range(4):
                autofocus.columnconfigure(col, weight=1)
            ttk.Radiobutton(autofocus, text="Semi Auto", value=AutofocusMode.SEMI.value, variable=self.af_mode_var, command=self._on_autofocus_mode_change).grid(row=0, column=0, columnspan=2, sticky="w")
            ttk.Radiobutton(autofocus, text="Full Auto", value=AutofocusMode.FULL.value, variable=self.af_mode_var, command=self._on_autofocus_mode_change).grid(row=0, column=2, columnspan=2, sticky="w")
            fields = [
                ("Half-range", self.af_scan_range_var, "af_range_entry"),
                ("Step", self.af_scan_step_var, "af_step_entry"),
                ("Speed %", self.af_speed_var, "af_speed_entry"),
                ("Settle s", self.af_settle_seconds_var, "af_settle_entry"),
                ("Sample s", self.af_sample_seconds_var, "af_sample_entry"),
                ("Near best", self.af_near_best_ratio_var, "af_near_best_entry"),
            ]
            for index, (label, variable, name) in enumerate(fields):
                row, col = 1 + index // 2, (index % 2) * 2
                ttk.Label(autofocus, text=label).grid(row=row, column=col, sticky="w")
                entry = ttk.Entry(autofocus, textvariable=variable, width=7)
                entry.grid(row=row, column=col + 1, sticky="ew", padx=2)
                setattr(self, name, entry)
            ttk.Button(autofocus, text="Start Autofocus", command=self._start_autofocus).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 2), padx=(0, 2))
            ttk.Button(autofocus, text="Stop Autofocus", command=self._stop_autofocus).grid(row=4, column=2, columnspan=2, sticky="ew", pady=(8, 2), padx=(2, 0))
            for index, variable in enumerate(
                [self.af_status_var, self.af_offset_var, self.af_score_var, self.af_best_var, self.af_best_offset_var, self.af_final_offset_var, self.af_confirm_score_var, self.af_sample_count_var],
                start=5,
            ):
                ttk.Label(autofocus, textvariable=variable, wraplength=300).grid(row=index, column=0, columnspan=4, sticky="w")

        self.stitching_panel = ttk.LabelFrame(right, text="Image Stitching Controls", padding=8)
        stitching = self.stitching_panel
        stitching.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            stitching.columnconfigure(col, weight=1)
        ttk.Label(stitching, text="Focus mode").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            stitching,
            text=StitchingFocusMode.SEMI.value,
            value=StitchingFocusMode.SEMI.value,
            variable=self.stitch_focus_mode_var,
            command=self._on_stitch_focus_mode_change,
        ).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            stitching,
            text=StitchingFocusMode.FULL.value,
            value=StitchingFocusMode.FULL.value,
            variable=self.stitch_focus_mode_var,
            command=self._on_stitch_focus_mode_change,
        ).grid(row=0, column=2, columnspan=2, sticky="w")
        ttk.Label(stitching, textvariable=self.stitch_corner_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(stitching, textvariable=self.stitch_plane_var, wraplength=300).grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Button(stitching, text="Record Corner", command=self._record_stitching_corner).grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=(6, 2))
        ttk.Button(stitching, text="Delete Last", command=self._delete_last_stitching_corner).grid(row=3, column=2, sticky="ew", padx=2, pady=(6, 2))
        ttk.Button(stitching, text="Clear", command=self._clear_stitching_corners).grid(row=3, column=3, sticky="ew", padx=2, pady=(6, 2))
        for index, (label, variable) in enumerate(
            [("Overlap %", self.stitch_overlap_var), ("Speed %", self.stitch_speed_var), ("Settle s", self.stitch_settle_seconds_var), ("Frames/tile", self.stitch_sample_frames_var)]
        ):
            row, col = 4 + index // 2, (index % 2) * 2
            ttk.Label(stitching, text=label).grid(row=row, column=col, sticky="w")
            ttk.Entry(stitching, textvariable=variable, width=7).grid(row=row, column=col + 1, sticky="ew", padx=2)
        ttk.Label(stitching, text="Output root").grid(row=6, column=0, sticky="w")
        ttk.Entry(stitching, textvariable=self.stitch_output_root_var, width=28).grid(row=6, column=1, columnspan=3, sticky="ew", padx=2)
        ttk.Label(stitching, textvariable=self.stitch_calibration_var, wraplength=300).grid(row=7, column=0, columnspan=4, sticky="w")
        ttk.Label(stitching, textvariable=self.stitch_plan_var, wraplength=300).grid(row=8, column=0, columnspan=4, sticky="w")
        ttk.Label(stitching, textvariable=self.stitch_preflight_var, wraplength=300).grid(row=9, column=0, columnspan=4, sticky="w")
        ttk.Button(stitching, text="Start Scan", command=self._start_stitching_scan).grid(row=10, column=0, columnspan=2, sticky="ew", padx=2, pady=(8, 2))
        ttk.Button(stitching, text="Stop Scan", command=self._stop_stitching_scan).grid(row=10, column=2, columnspan=2, sticky="ew", padx=2, pady=(8, 2))
        ttk.Label(stitching, textvariable=self.stitch_progress_var).grid(row=11, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(stitching, textvariable=self.stitch_output_var, wraplength=300).grid(row=12, column=0, columnspan=4, sticky="w")

        self.mission_panel = ttk.LabelFrame(right, text="Mission Log", padding=8)
        mission = self.mission_panel
        mission.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        mission.columnconfigure(0, weight=1)
        self.mission_log_text = tk.Text(
            mission,
            height=9,
            width=44,
            wrap="word",
            state="disabled",
            bg=MISSION_LOG_BACKGROUND_COLOR,
            fg=MISSION_LOG_FOREGROUND_COLOR,
            insertbackground=MISSION_LOG_FOREGROUND_COLOR,
            selectbackground="#145c22",
            relief="flat",
        )
        self.mission_log_text.grid(row=0, column=0, sticky="ew")
        ttk.Label(mission, textvariable=self.status_var, wraplength=350, foreground="#8a4b00").grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._append_mission_log("SYSTEM READY. Awaiting operator command.")

        camera_controls = ttk.LabelFrame(right, text="Camera Controls", padding=8)
        camera_controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            camera_controls.columnconfigure(col, weight=1)
        ttk.Button(camera_controls, text="Read Properties", command=self._read_camera_properties).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Button(camera_controls, text="Reduce Overexposure", command=self._reduce_overexposure).grid(row=0, column=2, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Checkbutton(camera_controls, text="Auto Exposure", variable=self.auto_exposure_var).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(camera_controls, text="Auto White Balance", variable=self.auto_wb_var).grid(row=1, column=2, columnspan=2, sticky="w")
        for index, (label, variable) in enumerate(
            [("Exposure", self.exposure_var), ("Gain", self.gain_var), ("Brightness", self.brightness_setting_var), ("Contrast", self.contrast_var), ("WB Temp", self.wb_temperature_var)]
        ):
            row, col = 2 + index // 2, (index % 2) * 2
            ttk.Label(camera_controls, text=label).grid(row=row, column=col, sticky="w")
            ttk.Entry(camera_controls, textvariable=variable, width=8).grid(row=row, column=col + 1, sticky="ew", padx=2)
        ttk.Button(camera_controls, text="Apply Settings", command=self._apply_camera_settings).grid(row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=(6, 2))
        ttk.Button(camera_controls, text="Reset Auto Mode", command=self._reset_auto_camera_mode).grid(row=5, column=2, columnspan=2, sticky="ew", padx=2, pady=(6, 2))
        ttk.Label(camera_controls, textvariable=self.camera_properties_var, wraplength=300).grid(row=6, column=0, columnspan=4, sticky="w")

        self._on_autofocus_mode_change()
        self._apply_mode_layout()

    def _update_body_scrollregion(self, _event=None) -> None:
        self.body_canvas.configure(scrollregion=self.body_canvas.bbox("all"))

    def _resize_body_canvas_window(self, event) -> None:
        requested_width = self.body_frame.winfo_reqwidth()
        self.body_canvas.itemconfigure(self._body_window_id, width=max(event.width, requested_width))
        self._update_body_scrollregion()

    def _bind_body_mousewheel(self, _event) -> None:
        self.body_canvas.bind_all("<MouseWheel>", self._on_body_mousewheel)
        self.body_canvas.bind_all("<Shift-MouseWheel>", self._on_body_shift_mousewheel)
        self.body_canvas.bind_all("<Button-4>", self._on_body_mousewheel)
        self.body_canvas.bind_all("<Button-5>", self._on_body_mousewheel)

    def _unbind_body_mousewheel(self, _event) -> None:
        self.body_canvas.unbind_all("<MouseWheel>")
        self.body_canvas.unbind_all("<Shift-MouseWheel>")
        self.body_canvas.unbind_all("<Button-4>")
        self.body_canvas.unbind_all("<Button-5>")

    def _on_body_mousewheel(self, event) -> str:
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            units = -1 if event.delta > 0 else 1
        self.body_canvas.yview_scroll(units, "units")
        return "break"

    def _on_body_shift_mousewheel(self, event) -> str:
        units = -1 if event.delta > 0 else 1
        self.body_canvas.xview_scroll(units, "units")
        return "break"

    def _bind_keys(self) -> None:
        for key in ("a", "d", "w", "s", "q", "e"):
            self.bind(f"<KeyPress-{key}>", lambda event, pressed=key: self._manual_key(event, pressed))
        self.bind("<space>", lambda _e: self._stop_all())
        self.bind("<Escape>", lambda _e: self._emergency_stop())

    def _manual_key(self, event, key: str) -> None:
        if should_ignore_axis_shortcut(getattr(event, "widget", None)):
            return
        mapped = manual_shortcut_mapping(key)
        if mapped is None:
            return
        axis, direction = mapped
        self._move(axis, direction)

    def _return_focus_to_root(self, event):
        if should_return_focus_to_root_on_enter(getattr(event, "widget", None)):
            self.focus_set()
            return "break"
        return None

    def _numeric_entry_keypress(self, event):
        if should_release_numeric_entry_focus(getattr(event, "char", ""), getattr(event, "keysym", "")):
            self.focus_set()
            return "break"
        return None

    def _current_mode(self) -> Mode:
        try:
            return Mode(self.mode_var.get())
        except ValueError:
            return Mode.MANUAL

    def _on_mode_change(self) -> None:
        mode = self._current_mode()
        self.current_mode_var.set(f"Current mode: {mode.value}")
        self._apply_mode_layout()
        LOG.info("mode switched: %s", mode.value)
        self._set_status(f"Mode switched to {mode.value}. 软件急停不能替代物理急停。")

    def _current_stitch_focus_mode(self) -> StitchingFocusMode:
        try:
            return StitchingFocusMode(self.stitch_focus_mode_var.get())
        except ValueError:
            return StitchingFocusMode.FULL

    def _on_stitch_focus_mode_change(self) -> None:
        mode = self._current_stitch_focus_mode()
        self._refresh_stitching_labels()
        self._record_mission_event("stitch_focus_mode", mode.value)

    def _apply_mode_layout(self) -> None:
        spec = mode_panel_spec(self._current_mode())
        autofocus_visible = "autofocus" in spec.visible_sections
        stitching_visible = "image_stitching" in spec.visible_sections
        if hasattr(self, "autofocus_panel"):
            if autofocus_visible:
                self.autofocus_panel.grid()
            else:
                self.autofocus_panel.grid_remove()
        if hasattr(self, "stitching_panel"):
            if stitching_visible:
                self.stitching_panel.grid()
            else:
                self.stitching_panel.grid_remove()
        if autofocus_visible:
            self.af_plot_frame.grid()
        else:
            self.af_plot_frame.grid_remove()
        if stitching_visible:
            self.stitch_plot_frame.grid()
        else:
            self.stitch_plot_frame.grid_remove()

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
        self._record_mission_event("motor_connecting", port)

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
            self._record_mission_event("motor_disconnected")
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
            self.focus_reference_frame = None
            self.camera_status_var.set(f"Camera: opened index {index} backend {backend}")
            self._auto_tune_exposure_for_focus()
            props = self.camera.get_camera_properties()
            self.exposure_var.set(str(props.get("exposure") or ""))
            self.gain_var.set(str(props.get("gain") or ""))
            self._set_status("Camera opened. Exposure auto-tuned for focus.")
            self._record_mission_event("camera_opened", f"index {index} backend {backend}")
        else:
            self.camera_status_var.set("Camera: no-camera mode")
            self._show_camera_placeholder()
            self._set_status("Camera could not be opened; GUI remains available.")

    def _close_camera(self) -> None:
        self.camera.close()
        self.focus_rois = None
        self.focus_reference_frame = None
        self.camera_status_var.set("Camera: closed")
        self._show_camera_placeholder()
        self._set_status("Camera closed.")
        self._record_mission_event("camera_closed")

    def _auto_tune_exposure_for_focus(self) -> None:
        if not self.camera.is_open:
            return
        samples: list[dict[str, float]] = []
        try:
            self._set_status("Camera exposure calibration in progress.")
            self._record_mission_event("camera_exposure_tuning_started")
            self.camera.set_auto_exposure(False)
            self.camera.set_camera_property("gain", 0)
            for exposure in exposure_tuning_candidates():
                self.camera.set_camera_property("exposure", exposure)
                time.sleep(0.08)
                scores: list[float] = []
                diagnostics: list[dict[str, float | bool]] = []
                for _ in range(4):
                    frame = self.camera.read_frame()
                    if self.focus_rois is None:
                        self.focus_rois = auto_select_rois(frame)
                    scores.append(float(calculate_focus_index(frame, self.focus_rois)))
                    diagnostics.append(brightness_diagnostics(frame))
                score = robust_representative(scores)
                mean_brightness = float(np.mean([float(item["mean_brightness"]) for item in diagnostics]))
                saturation = float(np.mean([float(item["frame_saturation"]) for item in diagnostics]))
                sample = {
                    "exposure": float(exposure),
                    "focus_score": score,
                    "mean_brightness": mean_brightness,
                    "saturation_fraction": saturation,
                }
                samples.append(sample)
                LOG.info("camera exposure tune sample: %s", sample)
            best = choose_best_exposure_result(samples)
            self.camera.set_camera_property("exposure", best["exposure"])
            self.camera.set_camera_property("gain", 0)
            self.focus_rois = None
            self.focus_reference_frame = None
            LOG.info("camera exposure tune selected: %s", best)
            self.camera_properties_var.set(
                f"Auto exposure tune: exposure={best['exposure']} focus={best['focus_score']:.2f} "
                f"saturation={best['saturation_fraction']:.3f}"
            )
            self._record_mission_event(
                "camera_exposure_tuning_completed",
                f"exposure={best['exposure']} focus={best['focus_score']:.2f} saturation={best['saturation_fraction']:.3f}",
            )
        except Exception as exc:
            LOG.exception("camera exposure auto-tune failed")
            self.camera_properties_var.set(f"Auto exposure tune failed: {exc}")
            self._record_mission_event("camera_exposure_tuning_failed", exc)

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

    def _start_stage_image_calibration(self) -> None:
        if self.autofocus.running or self.stitching.running:
            self._set_status("Cannot calibrate XY scale while autofocus or stitching is running.")
            return
        if not self.camera.is_open:
            self._set_status("Camera is not open; cannot calibrate XY scale.")
            return
        if not self.controller or not self.controller.is_open:
            self._set_status("Motor is not connected; cannot calibrate XY scale.")
            return
        confirmed = messagebox.askyesno(
            "Confirm XY Scale Calibration",
            "即将在当前位置附近短距离移动 X/Y 并拍摄临时帧，用于标定画面像素与电机 pulse 的关系。\n\n"
            "请确认当前位置安全、画面有足够纹理、物理急停可用。\n"
            "软件急停不能替代物理急停。\n\n是否继续？",
        )
        if not confirmed:
            self._set_status("XY scale calibration cancelled.")
            return
        self.stitching.running = True
        self.stitching.stop_requested = False
        self.stage_calibration_var.set("XY calibration: running")
        self._set_manual_controls_enabled(False)
        self.running_state_var.set("Running state: XY calibration")
        threading.Thread(target=self._stage_image_calibration_worker, daemon=True).start()

    def _stage_image_calibration_worker(self) -> None:
        start = dict(self.absolute_pos)
        try:
            calibration = None
            last_error = "no trial executed"
            for step in (80, 160, 320):
                self.device_queue.put(("stage_calibration_status", f"XY calibration: trying {step} pulse test move"))
                trial = self._build_current_position_trial(step)
                frames = []
                for label, point in (("reference", trial.reference), ("X", trial.x_trial), ("Y", trial.y_trial)):
                    self.device_queue.put(("stage_calibration_status", f"XY calibration: capturing {label} frame @ {step} pulses"))
                    self._move_to_absolute_position(point.x, point.y, point.z, 90)
                    self._sleep_with_stitching_stop(0.25)
                    frame, _score = self._capture_stable_stitching_frame(5)
                    frames.append(frame)
                try:
                    self.device_queue.put(("stage_calibration_status", f"XY calibration: estimating scale from {step} pulse move"))
                    calibration = estimate_calibration_from_frames(frames[0], frames[1], frames[2], trial, 0.0)
                    break
                except ValueError as exc:
                    last_error = str(exc)
                    self.device_queue.put(("stage_calibration_status", f"XY calibration: retry needed ({last_error})"))
                    LOG.warning("XY scale calibration retry required: step=%s error=%s", step, exc)
            self.device_queue.put(("stage_calibration_status", "XY calibration: returning to start position"))
            self._move_to_absolute_position(start["X"], start["Y"], start["Z"], 90)
            if calibration is None:
                raise RuntimeError(f"XY scale calibration failed: {last_error}")
            self.device_queue.put(("stage_calibration_done", calibration))
        except Exception as exc:
            LOG.exception("XY scale calibration failed")
            if self.controller and self.controller.is_open:
                try:
                    self.controller.stop_all()
                except Exception:
                    LOG.exception("stop_all after XY scale calibration failure failed")
            self.device_queue.put(("stage_calibration_error", f"XY scale calibration failed: {exc}"))

    def _build_current_position_trial(self, step: int):
        origin = TilePoint(
            row=-1,
            col=-1,
            x=int(self.absolute_pos["X"]),
            y=int(self.absolute_pos["Y"]),
            z=int(self.absolute_pos["Z"]),
        )
        x_trial = TilePoint(row=-1, col=-1, x=origin.x + int(step), y=origin.y, z=origin.z)
        y_trial = TilePoint(row=-1, col=-1, x=origin.x, y=origin.y + int(step), z=origin.z)
        from stitching_calibration import CalibrationTrialPlan

        return CalibrationTrialPlan(origin, x_trial, y_trial, int(step), int(step))

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
        if self.stitching.running:
            self._set_status("Image stitching is running; manual movement is disabled except Space/Esc.")
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

    def _on_preview_mouse_motion(self, event) -> None:
        if self.motion_control_mode_var.get() != "Mouse":
            self.preview_cursor_position = None
            return
        info = self.preview_display_info
        if info is None:
            return
        display_w, display_h, _frame_w, _frame_h = info
        if 0 <= event.x < display_w and 0 <= event.y < display_h:
            self.preview_cursor_position = (int(event.x), int(event.y))
        else:
            self.preview_cursor_position = None
        self._render_preview_overlay()

    def _on_preview_mouse_leave(self, _event) -> None:
        self.preview_cursor_position = None
        self._render_preview_overlay()

    def _on_preview_double_click(self, event) -> None:
        if self.motion_control_mode_var.get() != "Mouse":
            return
        if self.stage_image_calibration is None:
            self._set_status("XY scale is not calibrated; click Calibrate XY Scale first.")
            return
        info = self.preview_display_info
        if info is None:
            self._set_status("No live preview geometry is available for mouse movement.")
            return
        display_w, display_h, frame_w, frame_h = info
        if not (0 <= event.x < display_w and 0 <= event.y < display_h):
            return
        dx_frame = (float(event.x) - display_w / 2.0) * frame_w / display_w
        dy_frame = (float(event.y) - display_h / 2.0) * frame_h / display_h
        try:
            dx_pulses, dy_pulses = stage_delta_from_preview_delta(dx_frame, dy_frame, self.stage_image_calibration)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        if dx_pulses == 0 and dy_pulses == 0:
            self._set_status("Mouse target is already at the preview center.")
            return
        self._move_relative_xy(dx_pulses, dy_pulses, speed=90)

    def _move_relative_xy(self, dx_pulses: int, dy_pulses: int, speed: int = 90) -> None:
        if self.autofocus.running:
            self._set_status("Autofocus is running; mouse movement is disabled except Space/Esc.")
            return
        if self.stitching.running:
            self._set_status("Image stitching is running; mouse movement is disabled except Space/Esc.")
            return
        if not self.controller or not self.controller.is_open:
            self._set_status("Motor is not connected.")
            return
        self.recent_command_var.set(f"Recent command: Mouse move dX={dx_pulses} dY={dy_pulses} @ {speed}%")
        self.running_state_var.set("Running state: mouse move")

        def worker() -> None:
            try:
                for axis, delta in (("X", int(dx_pulses)), ("Y", int(dy_pulses))):
                    if delta == 0:
                        continue
                    direction = absolute_delta_to_controller_direction(axis, current=0, target=delta)
                    self.controller.move_relative(axis, direction, abs(delta), speed)
                positions = self._safe_read_positions(self.controller)
                self.device_queue.put(("positions", positions))
            except Exception as exc:
                LOG.exception("mouse move failed")
                self.device_queue.put(("error", f"Mouse move failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_all(self) -> None:
        self.recent_command_var.set("Recent command: Stop all axes")
        self._record_mission_event("controlled_stop")
        if self.autofocus.running:
            self.autofocus.stop_requested = True
            self._set_manual_controls_enabled(True)
            LOG.info("Stop Autofocus requested by Space/controlled stop")
            self.device_queue.put(("af_status", "AF status: stop requested"))
        if self.stitching.running:
            self.stitching.stop_requested = True
            self._set_manual_controls_enabled(True)
            LOG.info("Stop Stitching requested by Space/controlled stop")
            self.device_queue.put(("stitch_status", "Progress: stop requested"))
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
        self._record_mission_event("emergency_stop")
        if self.autofocus.running:
            self.autofocus.stop_requested = True
            self._set_manual_controls_enabled(True)
            LOG.warning("Emergency Stop during AF")
            self.device_queue.put(("af_status", "AF status: emergency stop requested"))
        if self.stitching.running:
            self.stitching.stop_requested = True
            self._set_manual_controls_enabled(True)
            LOG.warning("Emergency Stop during image stitching")
            self.device_queue.put(("stitch_status", "Progress: emergency stop requested"))
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

    def _record_stitching_corner(self) -> None:
        self.mode_var.set(Mode.IMAGE_STITCHING.value)
        self._on_mode_change()
        if self.stitching.running:
            self._set_status("Image stitching is running; cannot record corners now.")
            return
        missing = [axis for axis in ("X", "Y", "Z") if not self.position_available[axis]]
        if missing:
            self._set_status(f"Cannot record corner until positions are available: {', '.join(missing)}.")
            return
        corners = self.stitching.corners or []
        if len(corners) >= 4:
            self._set_status("Four stitching corners are already recorded. Delete or clear corners first.")
            return
        corner = SamplePlanePoint(
            label=f"c{len(corners) + 1}",
            x=int(self.absolute_pos["X"]),
            y=int(self.absolute_pos["Y"]),
            z=int(self.absolute_pos["Z"]),
        )
        corners.append(corner)
        self.stitching.corners = corners
        self._refresh_stitching_labels()
        self._draw_stitching_plane_view()
        self._set_status(f"Recorded stitching corner {corner.label}: X={corner.x} Y={corner.y} Z={corner.z}.")
        self._record_mission_event("stitch_corner_recorded", f"{corner.label} X={corner.x} Y={corner.y} Z={corner.z}")

    def _delete_last_stitching_corner(self) -> None:
        if self.stitching.running:
            self._set_status("Image stitching is running; cannot delete corners now.")
            return
        corners = self.stitching.corners or []
        if not corners:
            self._set_status("No stitching corners to delete.")
            return
        removed = corners.pop()
        self.stitching.corners = corners
        self._refresh_stitching_labels()
        self._draw_stitching_plane_view()
        self._set_status(f"Deleted stitching corner {removed.label}.")
        self._record_mission_event("stitch_corner_deleted", removed.label)

    def _clear_stitching_corners(self) -> None:
        if self.stitching.running:
            self._set_status("Image stitching is running; cannot clear corners now.")
            return
        self.stitching.corners = []
        self.stitching.planned_tiles = []
        self.stitching.current_tile_index = None
        self.stitching.captured_tile_count = 0
        self.stitching.calibration = None
        self.stitch_calibration_var.set("Calibration: pending")
        self.stitch_plan_var.set("Plan: pending")
        self._refresh_stitching_labels()
        self._draw_stitching_plane_view()
        self._set_status("Stitching corners cleared.")
        self._record_mission_event("stitch_corners_cleared")

    def _refresh_stitching_labels(self) -> None:
        corners = self.stitching.corners or []
        self.stitch_corner_var.set(f"Corners: {len(corners)}/4")
        if len(corners) == 4:
            if self._current_stitch_focus_mode() == StitchingFocusMode.SEMI:
                self.stitch_plane_var.set("Plane: will use manually focused corner Z values")
            else:
                self.stitch_plane_var.set("Plane: will autofocus a 3x3 probe grid and robustly fit when scan starts")
        elif len(corners) >= 1:
            self.stitch_plane_var.set("Plane: waiting for four XY corners")
        else:
            self.stitch_plane_var.set("Plane: not fitted")

    def _draw_stitching_plane_view(self) -> None:
        canvas = getattr(self, "stitch_plane_canvas", None)
        if canvas is None:
            return
        self._last_stitch_plot_time = time.monotonic()
        self._last_stitch_plot_progress = max(
            int(self.stitching.captured_tile_count or 0),
            int(self.stitching.current_tile_index or 0),
        )
        canvas.delete("all")
        width = max(10, canvas.winfo_width(), int(float(canvas.cget("width"))))
        height = max(10, canvas.winfo_height(), int(float(canvas.cget("height"))))
        margin = 18
        canvas.create_rectangle(0, 0, width, height, fill="#06110d", outline="")
        corners = self.stitching.corners or []
        tiles = self.stitching.planned_tiles or []
        if not corners and not tiles:
            canvas.create_text(10, 10, anchor="nw", fill="#7d9d8d", text="Record four XY corners to preview the scan area.")
            return
        display_corners = corners
        if len(corners) >= 3:
            boundary = boundary_polygon_from_points(corners)
            display_corners = [
                next(corner for corner in corners if (corner.x, corner.y) == coordinate)
                for coordinate in boundary
            ]
        model = build_stitching_view_model(
            display_corners,
            tiles,
            current_tile_index=self.stitching.current_tile_index,
            captured_tile_count=self.stitching.captured_tile_count,
        )

        def xy(point):
            x = margin + point.nx * max(1, width - margin * 2)
            y = margin + point.ny * max(1, height - margin * 2)
            return x, y

        x_spacing = sorted({point.nx for point in model.tiles})
        y_spacing = sorted({point.ny for point in model.tiles})

        def half_cell(values, span):
            gaps = [later - earlier for earlier, later in zip(values, values[1:]) if later > earlier]
            return (min(gaps) * span * 0.52) if gaps else (span * 0.46)

        plot_width = max(1, width - margin * 2)
        plot_height = max(1, height - margin * 2)
        half_width = half_cell(x_spacing, plot_width)
        half_height = half_cell(y_spacing, plot_height)
        draw_indices = preview_tile_draw_indices(
            len(model.tiles),
            current_tile_index=self.stitching.current_tile_index,
            captured_tile_count=self.stitching.captured_tile_count,
        )
        for index, point in enumerate(model.tiles):
            if index not in draw_indices:
                continue
            x, y = xy(point)
            fill = ""
            if point.state == "done":
                fill = "#0ea95c"
            elif point.state == "current":
                fill = "#39ff88"
            canvas.create_rectangle(
                x - half_width,
                y - half_height,
                x + half_width,
                y + half_height,
                fill=fill,
                outline="#39ff88",
                width=1,
            )
        if len(model.corners) >= 2:
            polygon = []
            for point in model.corners:
                polygon.extend(xy(point))
            if len(model.corners) >= 3:
                polygon.extend(xy(model.corners[0]))
            canvas.create_line(*polygon, fill="#5bffa0", width=1)
        for point in model.corners:
            x, y = xy(point)
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#06110d", outline="#5bffa0")
            canvas.create_text(x + 7, y - 7, anchor="w", fill="#b6fbd2", text=f"{point.label} z={point.z}")
        if len(model.tiles) > STITCH_PREVIEW_SAMPLE_TARGET:
            canvas.create_text(
                10,
                height - 10,
                anchor="sw",
                fill="#b6fbd2",
                text=f"Preview sampled {len(draw_indices)}/{len(model.tiles)} tiles",
            )

    def _request_stitching_plane_redraw(self, *, force: bool = False) -> None:
        tiles = self.stitching.planned_tiles or []
        if force or len(tiles) <= STITCH_PREVIEW_SAMPLE_TARGET:
            self._draw_stitching_plane_view()
            return
        progress = max(
            int(self.stitching.captured_tile_count or 0),
            int(self.stitching.current_tile_index or 0),
        )
        now = time.monotonic()
        if (
            progress == 0
            or progress >= len(tiles) - 1
            or progress - self._last_stitch_plot_progress >= LARGE_STITCH_REDRAW_TILE_INTERVAL
            or now - self._last_stitch_plot_time >= LARGE_STITCH_REDRAW_SECONDS
        ):
            self._draw_stitching_plane_view()

    def _read_stitching_params(self) -> tuple[float, int, float, int, Path]:
        try:
            overlap_percent = float(self.stitch_overlap_var.get())
            if not 0.0 <= overlap_percent < 90.0:
                raise ValueError("Overlap % must be between 0 and 90.")
            speed = max(1, min(100, int(float(self.stitch_speed_var.get()))))
            settle_seconds = max(0.0, float(self.stitch_settle_seconds_var.get()))
            sample_frames = max(1, int(float(self.stitch_sample_frames_var.get())))
        except ValueError as exc:
            if "Overlap" in str(exc):
                raise
            raise ValueError("Image stitching parameters must be numbers.") from exc
        output_root = Path(self.stitch_output_root_var.get()).expanduser()
        return overlap_percent, speed, settle_seconds, sample_frames, output_root

    def _start_stitching_scan(self) -> None:
        self.mode_var.set(Mode.IMAGE_STITCHING.value)
        self._on_mode_change()
        if self.stitching.running:
            self._set_status("Image stitching scan is already running.")
            return
        if not self.camera.is_open:
            self._set_status("Camera is not open; cannot start image stitching.")
            return
        if not self.controller or not self.controller.is_open:
            self._set_status("Motor is not connected; cannot start image stitching.")
            return
        corners = list(self.stitching.corners or [])
        if len(corners) != 4:
            self._set_status("Record exactly four stitching XY corners before scanning.")
            return
        try:
            overlap_percent, speed, settle_seconds, sample_frames, output_root = self._read_stitching_params()
            bounds = bounds_from_plane_points(corners)
        except Exception as exc:
            self._set_status(f"Image stitching setup failed: {exc}")
            return
        focus_mode = self._current_stitch_focus_mode()
        preflight_report = format_stitching_preflight_report(
            corner_count=len(corners),
            has_calibration=self.stage_image_calibration is not None,
            last_focus_info=self.last_focus_info,
            output_root=str(output_root),
            overlap_percent=overlap_percent,
            sample_frames=sample_frames,
        )
        self.stitch_preflight_var.set(preflight_report)
        self._last_stitching_preflight_report = preflight_report
        self._set_status(preflight_report)
        self._record_mission_event("stitch_preflight", f"{preflight_report}; focus mode {focus_mode.value}")
        focus_prompt = (
            "半自动模式：将使用用户手动对焦并记录的四个角点 Z 值拟合样品平面，不会在扫描开始前重对焦。\n"
            if focus_mode == StitchingFocusMode.SEMI
            else "全自动模式：将四个记录点作为边界，在区域内自动对焦 3x3 probe 点并鲁棒拟合样品平面。\n"
        )
        confirmed = messagebox.askyesno(
            "Confirm Image Stitching Scan",
            f"{focus_prompt}随后自动规划并拍摄拼场图片。\n\n"
            f"Requested overlap = {overlap_percent:.1f}%\n"
            f"{preflight_report}\n"
            "请确认扫描范围安全、物理急停可用、没有探针会碰撞。\n"
            "软件急停不能替代物理急停。\n\n是否继续？",
        )
        if not confirmed:
            self._set_status("Image stitching scan cancelled.")
            self._record_mission_event("stitch_cancelled")
            return

        self.stitching.running = True
        self.stitching.stop_requested = False
        self.stitching.planned_tiles = []
        self.stitching.current_tile_index = None
        self.stitching.captured_tile_count = 0
        self.stitching.calibration = None
        self._draw_stitching_plane_view()
        self._set_manual_controls_enabled(False)
        self.running_state_var.set("Running state: image stitching")
        self.recent_command_var.set("Recent command: Start Image Stitching")
        self.stitch_progress_var.set("Progress: calibrating inside four-corner boundary")
        self.stitch_calibration_var.set("Calibration: running")
        self.stitch_plan_var.set("Plan: waiting for calibration")
        LOG.info("Start Image Stitching: overlap=%.1f speed=%s", overlap_percent, speed)
        self._record_mission_event("stitch_started", f"focus mode {focus_mode.value}, overlap {overlap_percent:.1f}%")
        threading.Thread(
            target=self._stitching_scan_worker,
            args=(corners, None, bounds, overlap_percent, speed, settle_seconds, sample_frames, output_root, focus_mode.value),
            daemon=True,
        ).start()

    def _stop_stitching_scan(self) -> None:
        if not self.stitching.running:
            self._set_status("Image stitching scan is not running.")
            return
        self.stitching.stop_requested = True
        self.stitch_progress_var.set("Progress: stop requested")
        self._set_status("Stopping image stitching scan...")
        self._record_mission_event("stitch_stop_requested")
        if self.controller and self.controller.is_open:
            threading.Thread(target=self._safe_stop_worker, daemon=True).start()

    def _stitching_scan_worker(
        self,
        corners: list[SamplePlanePoint],
        plane,
        bounds,
        overlap_percent: float,
        speed: int,
        settle_seconds: float,
        sample_frames: int,
        output_root: Path,
        focus_mode: str = StitchingFocusMode.FULL.value,
    ) -> None:
        saved_tiles: list[TileRecord] = []
        store: StitchingSessionStore | None = None
        mosaic_builder: IncrementalMosaicBuilder | None = None
        mosaic_executor: ThreadPoolExecutor | None = None
        mosaic_futures: list[Future] = []
        try:
            store = StitchingSessionStore.create(output_root)
            self.device_queue.put(("stitch_output", f"Output: {store.path}"))
            original_corners = list(corners)
            selected_focus_mode = StitchingFocusMode(focus_mode)
            if selected_focus_mode == StitchingFocusMode.SEMI:
                self.device_queue.put(("mission_event", ("stitch_semiauto_plane", None)))
                focused_corners = original_corners
                plane_report = evaluate_plane_consistency(
                    focused_corners,
                    max_confirmation_residual=STITCHING_PLANE_CONFIRMATION_RESIDUAL_PULSES,
                )
                plane = plane_report.plane
                plane_validation = plane_report.to_dict()
                self.device_queue.put(("stitch_status", f"Progress: {plane_report.message}"))
            else:
                self.device_queue.put(("mission_event", ("stitch_fullauto_plane", None)))
                seed_plane = fit_sample_plane(original_corners)
                probe_points = generate_plane_probe_points(original_corners, seed_plane, grid_size=3)
                focused_probes = []
                for index, probe in enumerate(probe_points, start=1):
                    self._raise_if_stitching_stopped()
                    self.device_queue.put(("stitch_status", f"Progress: autofocus probe {index}/{len(probe_points)}"))
                    focused_probes.append(self._focus_stitching_corner(probe, speed, settle_seconds, sample_frames))
                robust_report = fit_sample_plane_robust(
                    focused_probes,
                    max_abs_residual=STITCHING_PLANE_CONFIRMATION_RESIDUAL_PULSES,
                    min_inliers=max(6, len(focused_probes) - 2),
                )
                plane = robust_report.plane
                focused_corners = original_corners
                plane_validation = robust_report.to_dict()
                self.device_queue.put(("stitch_status", f"Progress: {robust_report.message}"))
            corners = original_corners
            bounds = bounds_from_plane_points(corners)
            boundary_points = boundary_polygon_from_points(corners)
            calibration = (
                replace(self.stage_image_calibration, overlap_percent=overlap_percent)
                if self.stage_image_calibration is not None
                else None
            )
            if calibration is None:
                used_steps: set[tuple[int, int]] = set()
                calibration_error = "no trial executed"
                for requested_step in (40, 80, 160, 320):
                    trial = build_in_bounds_trial_plan(
                        bounds,
                        plane,
                        boundary_points=boundary_points,
                        preferred_step_pulses=requested_step,
                    )
                    step_key = (trial.x_step_pulses, trial.y_step_pulses)
                    if step_key in used_steps:
                        continue
                    used_steps.add(step_key)
                    trial_frames = []
                    for label, trial_point in (
                        ("reference", trial.reference),
                        ("X", trial.x_trial),
                        ("Y", trial.y_trial),
                    ):
                        self._raise_if_stitching_stopped()
                        self.device_queue.put(
                            ("stitch_status", f"Progress: calibrating {label} frame ({step_key[0]}/{step_key[1]} pulses)")
                        )
                        self._move_to_absolute_position(trial_point.x, trial_point.y, trial_point.z, speed)
                        self._sleep_with_stitching_stop(settle_seconds)
                        frame, _focus_score = self._capture_stable_stitching_frame(sample_frames)
                        trial_frames.append(frame)
                    try:
                        calibration = estimate_calibration_from_frames(
                            trial_frames[0], trial_frames[1], trial_frames[2], trial, overlap_percent
                        )
                        break
                    except ValueError as exc:
                        calibration_error = str(exc)
                        LOG.warning("stitching calibration retry required: step=%s error=%s", step_key, exc)
                if calibration is None:
                    raise RuntimeError(f"automatic calibration failed inside recorded boundary: {calibration_error}")
            plan = generate_overlap_scan_plan(
                bounds,
                plane,
                frame_width=calibration.frame_width,
                frame_height=calibration.frame_height,
                x_pixels_per_pulse=calibration.x_pixels_per_pulse,
                y_pixels_per_pulse=calibration.y_pixels_per_pulse,
                overlap_percent=overlap_percent,
                boundary_points=boundary_points,
            )
            tiles = plan.tiles
            planned_records = [
                TileRecord(row=tile.row, col=tile.col, x=tile.x, y=tile.y, z=tile.z, filename="", focus_score=0.0)
                for tile in tiles
            ]
            self.device_queue.put(("stitch_plan", (calibration, plan)))
            for index, tile in enumerate(tiles, start=1):
                self._raise_if_stitching_stopped()
                self.device_queue.put(("stitch_tile_index", index - 1))
                self.device_queue.put(("stitch_status", f"Progress: moving tile {index}/{len(tiles)}"))
                self._move_to_absolute_position(tile.x, tile.y, tile.z, speed)
                self._sleep_with_stitching_stop(settle_seconds)
                frame, focus_score = self._capture_stable_stitching_frame(sample_frames)
                capture_info = getattr(self, "_last_stitching_capture_info", {})
                record = TileRecord(
                    row=tile.row,
                    col=tile.col,
                    x=tile.x,
                    y=tile.y,
                    z=tile.z,
                    filename="",
                    focus_score=focus_score,
                    mean_brightness=capture_info.get("mean_brightness"),
                    saturation_fraction=capture_info.get("saturation_fraction"),
                    underexposed_fraction=capture_info.get("underexposed_fraction"),
                    overexposed=capture_info.get("overexposed"),
                    sample_frames=capture_info.get("sample_frames"),
                    saved_from_raw_frame=capture_info.get("saved_from_raw_frame"),
                )
                saved = store.save_tile(frame, record)
                saved_tiles.append(saved)
                if mosaic_builder is None:
                    mosaic_builder = IncrementalMosaicBuilder(
                        planned_records,
                        frame.shape,
                        x_pixels_per_pulse=calibration.x_pixels_per_pulse,
                        y_pixels_per_pulse=calibration.y_pixels_per_pulse,
                    )
                    mosaic_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stitch-mosaic")
                    self.device_queue.put(("stitch_status", "Progress: incremental mosaic worker started"))
                    self.device_queue.put(("mission_event", ("stitch_assembling", "incremental")))
                mosaic_futures.append(mosaic_executor.submit(mosaic_builder.add_tile, index - 1, frame))
                self.device_queue.put(("stitch_tile_saved", index))
                self.device_queue.put(("stitch_status", f"Progress: saved tile {index}/{len(tiles)}"))
            store.write_metadata(
                corners=corners,
                tiles=saved_tiles,
                settings={
                    "rows": plan.rows,
                    "cols": plan.cols,
                    "speed": speed,
                    "settle_seconds": settle_seconds,
                    "sample_frames": sample_frames,
                    "overlap_percent": overlap_percent,
                    "preflight": getattr(self, "_last_stitching_preflight_report", ""),
                    "focus_mode": selected_focus_mode.value,
                    "plane_validation": plane_validation,
                },
                plane=plane,
                calibration=calibration,
            )
            store.write_tile_quality_csv(saved_tiles)
            self.device_queue.put(("mission_event", ("stitch_scan_completed", None)))
            self.device_queue.put(("stitch_status", "Progress: finalizing stitched mosaic..."))
            if mosaic_builder is not None:
                self.device_queue.put(("stitch_status", "Progress: waiting for mosaic blending worker..."))
                for future in mosaic_futures:
                    future.result()
                self.device_queue.put(("stitch_status", "Progress: writing mechanical-only debug mosaic..."))
                mosaic_builder.write(store.path / "mechanical_only_mosaic.png")
                self.device_queue.put(("stitch_status", "Progress: writing mosaic boundary debug image..."))
                mosaic_builder.write_with_boundaries(store.path / "mosaic_with_boundaries.png")
                self.device_queue.put(("stitch_status", "Progress: writing stitched mosaic image..."))
                mosaic_path = mosaic_builder.write(store.path / "stitched_mosaic.png")
            else:
                self.device_queue.put(("mission_event", ("stitch_assembling", None)))
                self.device_queue.put(("stitch_status", "Progress: assembling saved tiles into mosaic..."))
                mosaic_path = stitch_session_by_metadata(store.path)
            self.device_queue.put(("stitch_done", (store.path, mosaic_path)))
        except InterruptedError:
            if store is not None and saved_tiles:
                store.write_metadata(
                    corners=corners,
                    tiles=saved_tiles,
                    settings={"stopped": True, "overlap_percent": overlap_percent},
                    plane=plane,
                )
            self.device_queue.put(("stitch_done", (store.path if store else None, None)))
        except Exception as exc:
            LOG.exception("image stitching scan failed")
            if self.controller and self.controller.is_open:
                try:
                    self.controller.stop_all()
                except Exception:
                    LOG.exception("stop_all after image stitching failure failed")
            self.device_queue.put(("stitch_error", f"Image stitching failed: {exc}"))
        finally:
            if mosaic_executor is not None:
                mosaic_executor.shutdown(wait=True, cancel_futures=True)
            if mosaic_builder is not None:
                mosaic_builder.close()

    def _focus_stitching_corner(
        self,
        corner: SamplePlanePoint,
        speed: int,
        settle_seconds: float,
        sample_frames: int,
    ) -> SamplePlanePoint:
        self._move_to_absolute_position(corner.x, corner.y, corner.z, speed)
        params = AutofocusParams(
            scan_range=50,
            scan_step=5,
            autofocus_speed=max(1, min(100, speed)),
            settle_seconds=max(0.1, settle_seconds),
            sample_seconds=max(MIN_STITCHING_CORNER_AUTOFOCUS_SAMPLE_SECONDS, sample_frames * 0.08),
            near_best_ratio=0.96,
        )
        try:
            self._inline_autofocus_status_prefix = f"Progress: autofocus corner {corner.label}"
            final_z, _score = self._run_inline_autofocus(params, AutofocusMode.FULL)
        except RuntimeError as exc:
            if "not enough valid focus frames" in str(exc):
                raise RuntimeError(
                    f"corner {corner.label} autofocus could not collect enough camera frames; "
                    "check the live camera stream or increase Settle s"
                ) from exc
            raise
        finally:
            self._inline_autofocus_status_prefix = "Progress: autofocus"
        return SamplePlanePoint(corner.label, corner.x, corner.y, final_z)

    def _run_inline_autofocus(
        self,
        params: AutofocusParams,
        autofocus_mode: AutofocusMode,
        *,
        status_prefix: str | None = None,
    ) -> tuple[int, float]:
        if status_prefix is None:
            status_prefix = getattr(self, "_inline_autofocus_status_prefix", "Progress: autofocus")
        current_offset = 0
        points: list[AutofocusSamplePoint] = []
        self._sleep_with_stitching_stop(params.settle_seconds)
        self.device_queue.put(("stitch_status", f"{status_prefix}: baseline"))
        baseline_score, baseline_iqr, baseline_count = self.sample_focus_at_current_z(params.sample_seconds)
        points.append(AutofocusSamplePoint(0, baseline_score, baseline_iqr, baseline_count))
        center_offset = 0
        for pass_plan in build_autofocus_pass_plan(params, autofocus_mode):
            pass_points: list[AutofocusSamplePoint] = []
            offsets = [center_offset + item for item in build_scan_offsets(pass_plan.scan_range, pass_plan.scan_step)]
            self.device_queue.put(
                (
                    "stitch_status",
                    f"{status_prefix}: {pass_plan.name} pass step {pass_plan.scan_step} ({len(offsets)} samples)",
                )
            )
            for offset in offsets:
                self._raise_if_stitching_stopped()
                delta = offset - current_offset
                if delta:
                    self._move_z_for_inline_operation(delta, params.autofocus_speed)
                    current_offset = offset
                self._sleep_with_stitching_stop(params.settle_seconds)
                score, iqr, frame_count = self.sample_focus_at_current_z(params.sample_seconds)
                point = AutofocusSamplePoint(offset, score, iqr, frame_count)
                points.append(point)
                pass_points.append(point)
            center_offset = select_final_autofocus_point(pass_points, params.near_best_ratio).offset
        final_point = select_final_autofocus_point(points, params.near_best_ratio)
        delta_back = final_point.offset - current_offset
        if delta_back:
            self._move_z_for_inline_operation(delta_back, params.autofocus_speed)
        return int(self.absolute_pos["Z"]), final_point.score

    def _move_z_for_inline_operation(self, delta: int, speed: int) -> None:
        if delta == 0:
            return
        self._raise_if_stitching_stopped()
        direction = 1 if delta > 0 else -1
        controller_direction = logical_direction_to_controller_direction("Z", direction)
        self.controller.move_relative("Z", controller_direction, abs(delta), speed)
        self.absolute_pos["Z"] = int(self.absolute_pos["Z"]) + int(delta)

    def _move_to_absolute_position(self, x: int, y: int, z: int, speed: int) -> None:
        for axis, target in (("Z", z), ("X", x), ("Y", y)):
            self._raise_if_stitching_stopped()
            current = int(self.absolute_pos[axis])
            delta = int(target) - current
            if delta == 0:
                continue
            controller_direction = absolute_delta_to_controller_direction(axis, current=current, target=target)
            self.controller.move_relative(axis, controller_direction, abs(delta), speed)
            self.absolute_pos[axis] = int(target)

    def _capture_stable_stitching_frame(self, sample_frames: int):
        best_frame = None
        best_score = float("-inf")
        best_info = None
        local_rois = None
        reference_frame = None
        for _ in range(sample_frames):
            self._raise_if_stitching_stopped()
            with self.camera_lock:
                frame = self.camera.read_frame()
            if reference_frame is None:
                reference_frame = frame.copy()
            stabilized_frame, _motion_info = stabilize_frame_translation(reference_frame, frame)
            if local_rois is None:
                local_rois = auto_select_rois(stabilized_frame)
            info = brightness_diagnostics(stabilized_frame)
            focus_score = float(calculate_focus_index(stabilized_frame, local_rois))
            if bool(info.get("overexposed", False)):
                focus_score *= 0.75
            if focus_score > best_score:
                best_score = focus_score
                best_frame = frame.copy()
                best_info = dict(info)
            time.sleep(0.03)
        if best_frame is None:
            raise RuntimeError("no valid camera frame captured for stitching")
        if best_info is None:
            best_info = brightness_diagnostics(best_frame)
        self._last_stitching_capture_info = {
            "mean_brightness": float(best_info.get("mean_brightness", 0.0)),
            "saturation_fraction": float(best_info.get("frame_saturation", 0.0)),
            "underexposed_fraction": float(best_info.get("underexposed_fraction", 0.0)),
            "overexposed": bool(best_info.get("overexposed", False)),
            "sample_frames": int(sample_frames),
            "saved_from_raw_frame": True,
        }
        return best_frame, best_score

    def _sleep_with_stitching_stop(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._raise_if_stitching_stopped()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _raise_if_stitching_stopped(self) -> None:
        if self.stitching.stop_requested:
            raise InterruptedError("image stitching stopped")

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
            f"Half-range = {params.scan_range} pulses，表示从当前位置到任一方向边缘的距离；总扫描宽度约为 {params.scan_range * 2} pulses。\n"
            "请确认样品/探针/镜头安全。\n"
            "请确认物理急停可用。\n"
            "默认 speed = 90；第一次测试仍请确认 Z 轴移动路径安全。\n"
            "软件急停不能替代物理急停。\n"
            "是否继续？",
        )
        if not confirmed:
            self._set_status("Autofocus cancelled.")
            return

        self.autofocus.reset()
        self.autofocus.running = True
        self._set_manual_controls_enabled(False)
        self.running_state_var.set("Running state: autofocus")
        self.recent_command_var.set("Recent command: Start Autofocus")
        self.af_status_var.set("AF status: running")
        self._refresh_autofocus_labels()
        self._draw_autofocus_plot()
        LOG.info("Start Autofocus")
        LOG.info("Autofocus parameters: %s", params)
        self._record_mission_event("autofocus_started")
        threading.Thread(target=self._autofocus_worker, args=(params, self._current_autofocus_mode()), daemon=True).start()

    def _stop_autofocus(self) -> None:
        if not self.autofocus.running:
            self._set_status("Autofocus is not running.")
            return
        self.autofocus.stop_requested = True
        LOG.info("Stop Autofocus")
        self.af_status_var.set("AF status: stop requested")
        self._set_status("Stopping autofocus...")
        self._record_mission_event("autofocus_stopped")
        if self.controller and self.controller.is_open:
            threading.Thread(target=self._safe_stop_worker, daemon=True).start()

    def _current_autofocus_mode(self) -> AutofocusMode:
        try:
            return AutofocusMode(self.af_mode_var.get())
        except ValueError:
            return AutofocusMode.SEMI

    def _on_autofocus_mode_change(self) -> None:
        full_auto = self._current_autofocus_mode() == AutofocusMode.FULL
        state = "disabled" if full_auto else "normal"
        for entry_name in (
            "af_step_entry",
            "af_speed_entry",
            "af_settle_entry",
            "af_sample_entry",
            "af_near_best_entry",
        ):
            entry = getattr(self, entry_name, None)
            if entry is not None:
                entry.configure(state=state)
        if full_auto:
            self.af_status_var.set("AF status: Full Auto uses range only; step/other parameters are chosen automatically.")
        else:
            self.af_status_var.set("AF status: Semi Auto uses user range and step.")

    def _set_manual_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in getattr(self, "manual_move_buttons", []):
            button.configure(state=state)

    def _read_autofocus_params(self) -> AutofocusParams:
        try:
            if self._current_autofocus_mode() == AutofocusMode.FULL:
                return AutofocusParams(
                    scan_range=max(1, int(float(self.af_scan_range_var.get()))),
                    scan_step=5,
                    autofocus_speed=90,
                    settle_seconds=0.5,
                    sample_seconds=1.5,
                    near_best_ratio=0.96,
                )
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

    def _autofocus_worker(self, params: AutofocusParams, autofocus_mode: AutofocusMode = AutofocusMode.SEMI) -> None:
        current_offset = 0
        points: list[AutofocusSamplePoint] = []
        try:
            self.device_queue.put(("af_status", "AF status: baseline sampling"))
            self._sleep_with_autofocus_stop(params.settle_seconds)
            baseline_score, baseline_iqr, baseline_count = self.sample_focus_at_current_z(params.sample_seconds)
            baseline = AutofocusSamplePoint(0, baseline_score, baseline_iqr, baseline_count)
            points.append(baseline)
            self.autofocus.current_score = baseline_score
            LOG.info("Autofocus baseline: score=%.3f iqr=%.3f frames=%s", baseline_score, baseline_iqr, baseline_count)
            self.device_queue.put(("af_point", baseline))

            center_offset = 0
            for pass_plan in build_autofocus_pass_plan(params, autofocus_mode):
                self._raise_if_autofocus_stopped()
                self.device_queue.put(("af_status", f"AF status: {pass_plan.name} scan"))
                pass_points: list[AutofocusSamplePoint] = []
                raw_offsets = build_scan_offsets(pass_plan.scan_range, pass_plan.scan_step)
                offsets = [center_offset + offset for offset in raw_offsets]

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
                    self.autofocus.current_score = score
                    LOG.info(
                        "Autofocus %s point: offset=%s score=%.3f iqr=%.3f frames=%s",
                        pass_plan.name,
                        offset,
                        score,
                        iqr,
                        frame_count,
                    )
                    points.append(point)
                    pass_points.append(point)
                    self.device_queue.put(("af_point", point))

                center_offset = select_final_autofocus_point(pass_points, params.near_best_ratio).offset
                LOG.info("Autofocus %s best center=%s", pass_plan.name, center_offset)

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
                        "自动对焦完成，但确认分数偏低，建议重试或手动调整焦点。",
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
        if self.autofocus.stop_requested or self.stitching.stop_requested:
            raise InterruptedError("autofocus stopped")

    def sample_focus_at_current_z(self, sample_seconds: float) -> tuple[float, float, int]:
        values: list[float] = []
        failures = 0
        reference_frame = None
        deadline = time.monotonic() + sample_seconds
        while time.monotonic() < deadline:
            self._raise_if_autofocus_stopped()
            try:
                with self.camera_lock:
                    frame = self.camera.read_frame()
                    if reference_frame is None:
                        reference_frame = frame.copy()
                    stabilized_frame, motion_info = stabilize_frame_translation(reference_frame, frame)
                    if self.focus_rois is None:
                        self.focus_rois = auto_select_rois(stabilized_frame)
                    focus_index = calculate_focus_index(stabilized_frame, self.focus_rois)
                    LOG.debug("autofocus stabilization: %s", motion_info)
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

    def _read_speed(self, default: int = 90) -> int:
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
        if (
            self.controller
            and self.controller.is_open
            and not self.autofocus.running
            and not self.stitching.running
            and not self._position_poll_running
        ):
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
                    if self.focus_reference_frame is None:
                        self.focus_reference_frame = frame.copy()
                    stabilized_frame, motion_info = stabilize_frame_translation(self.focus_reference_frame, frame)
                    if self.focus_rois is None:
                        self.focus_rois = auto_select_rois(stabilized_frame)
                    focus_index = calculate_focus_index(stabilized_frame, self.focus_rois)
                    self.last_focus_info = brightness_diagnostics(stabilized_frame)
                    LOG.debug("live stabilization: %s", motion_info)
                self._update_focus(focus_index)
                self._update_focus_diagnostics(self.last_focus_info)
                self._show_frame(stabilized_frame)
            except Exception:
                LOG.exception("camera loop error")
                self.camera_status_var.set("Camera: read error, no-camera mode")
                self.camera.close()
                self._show_camera_placeholder("CAMERA FEED LOST\nCheck camera connection")
        self._schedule_after(60, self._camera_loop)

    def _update_focus(self, focus_index: float) -> None:
        self.focus_var.set(f"Sharpness index: {focus_index:.2f}")

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
            self._show_camera_placeholder("CAMERA UNAVAILABLE\nOpenCV is not installed")
            return

        now = time.monotonic()
        if now - self._last_frame_time < 0.04:
            return
        self._last_frame_time = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_w = VIDEO_PREVIEW_MAX_WIDTH
        label_h = VIDEO_PREVIEW_MAX_HEIGHT
        height, width = rgb.shape[:2]
        scale = min(label_w / width, label_h / height)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.preview_display_info = (new_w, new_h, width, height)
        self._preview_base_rgb = resized
        self._render_preview_overlay()

    def _render_preview_overlay(self) -> None:
        if self._preview_base_rgb is None or self.preview_display_info is None:
            return
        new_w, new_h, _width, _height = self.preview_display_info
        cursor = self.preview_cursor_position if self.motion_control_mode_var.get() == "Mouse" else None
        resized = draw_preview_crosshair(self._preview_base_rgb, cursor_position=cursor, include_center=True, cursor_radius=8)
        header = f"P6\n{new_w} {new_h}\n255\n".encode("ascii")
        ppm = header + np.ascontiguousarray(resized).tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        self.video_label.configure(image=self._photo, text="")

    def _show_camera_placeholder(self, text: str = PREVIEW_PLACEHOLDER_TEXT) -> None:
        self._placeholder_photo = tk.PhotoImage(width=VIDEO_PREVIEW_MAX_WIDTH, height=VIDEO_PREVIEW_MAX_HEIGHT)
        self._placeholder_photo.put(
            PREVIEW_BACKGROUND_COLOR,
            to=(0, 0, VIDEO_PREVIEW_MAX_WIDTH, VIDEO_PREVIEW_MAX_HEIGHT),
        )
        self.preview_display_info = None
        self._preview_base_rgb = None
        self.video_label.configure(image=self._placeholder_photo, text=text, compound="center")

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
                    self._record_mission_event("motor_connected")
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
                elif kind == "mission_event":
                    event_kind, event_payload = payload
                    self._record_mission_event(str(event_kind), event_payload)
                elif kind == "stage_calibration_status":
                    message = str(payload)
                    self.stage_calibration_var.set(message)
                    self._set_status(message)
                elif kind == "stage_calibration_done":
                    calibration = payload
                    self.stage_image_calibration = calibration
                    self.stitching.running = False
                    self.stitching.stop_requested = False
                    self._set_manual_controls_enabled(True)
                    self.running_state_var.set("Running state: idle")
                    self.stage_calibration_var.set(
                        "XY calibration: "
                        f"X={calibration.x_pixels_per_pulse:.4f} px/pulse  "
                        f"Y={calibration.y_pixels_per_pulse:.4f} px/pulse  "
                        f"confidence={calibration.confidence:.2f}"
                    )
                    self._set_status("XY scale calibration completed.")
                elif kind == "stage_calibration_error":
                    self.stitching.running = False
                    self.stitching.stop_requested = False
                    self._set_manual_controls_enabled(True)
                    self.running_state_var.set("Running state: error")
                    self.stage_calibration_var.set("XY calibration: failed")
                    self.recent_error_var.set(f"Recent error: {payload}")
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
                    self._set_manual_controls_enabled(True)
                    self.running_state_var.set("Running state: idle")
                    self.af_status_var.set(f"AF status: {payload}")
                    self._set_status(str(payload))
                    if "completed" in str(payload).lower():
                        self._record_mission_event("autofocus_completed")
                    else:
                        self._record_mission_event("autofocus_stopped")
                elif kind == "af_error":
                    self.autofocus.running = False
                    self.autofocus.stop_requested = False
                    self._set_manual_controls_enabled(True)
                    self.running_state_var.set("Running state: error")
                    self.recent_error_var.set(f"Recent error: {payload}")
                    self.af_status_var.set("AF status: failed")
                    self._set_status(str(payload))
                    self._record_mission_event("autofocus_failed", payload)
                elif kind == "stitch_status":
                    self.stitch_progress_var.set(str(payload))
                elif kind == "stitch_plan":
                    calibration, plan = payload
                    self.stitching.calibration = calibration
                    self.stitching.planned_tiles = plan.tiles
                    self.stitching.current_tile_index = None
                    self.stitching.captured_tile_count = 0
                    self.stitch_calibration_var.set(
                        "Calibration: "
                        f"X={calibration.x_pixels_per_pulse:.4f} px/pulse  "
                        f"Y={calibration.y_pixels_per_pulse:.4f} px/pulse  "
                        f"confidence={calibration.confidence:.2f}"
                    )
                    self.stitch_plan_var.set(
                        f"Plan: {plan.rows} rows x {plan.cols} cols = {len(plan.tiles)} tiles, "
                        f"overlap={plan.overlap_percent:.1f}%"
                    )
                    self._request_stitching_plane_redraw(force=True)
                elif kind == "stitch_tile_index":
                    self.stitching.current_tile_index = int(payload)
                    self._request_stitching_plane_redraw()
                elif kind == "stitch_tile_saved":
                    self.stitching.captured_tile_count = int(payload)
                    self._request_stitching_plane_redraw()
                elif kind == "stitch_output":
                    self.stitch_output_var.set(str(payload))
                elif kind == "stitch_done":
                    session_path, mosaic_path = payload
                    self.stitching.running = False
                    self.stitching.stop_requested = False
                    self._set_manual_controls_enabled(True)
                    self.running_state_var.set("Running state: idle")
                    if session_path is not None:
                        self.stitching.last_session_path = Path(session_path)
                    if mosaic_path is not None:
                        self.stitching.current_tile_index = None
                        self.stitching.captured_tile_count = len(self.stitching.planned_tiles or [])
                        self.stitching.last_mosaic_path = Path(mosaic_path)
                        self.stitch_output_var.set(f"Output: {mosaic_path}")
                        self.stitch_progress_var.set("Progress: completed")
                        self._set_status(f"Image stitching completed: {mosaic_path}")
                        self._record_mission_event("stitch_completed", mosaic_path)
                    else:
                        self.stitching.current_tile_index = None
                        self.stitch_progress_var.set("Progress: stopped")
                        self._set_status("Image stitching stopped.")
                    self._request_stitching_plane_redraw(force=True)
                elif kind == "stitch_error":
                    self.stitching.running = False
                    self.stitching.stop_requested = False
                    self.stitching.current_tile_index = None
                    self._set_manual_controls_enabled(True)
                    self.running_state_var.set("Running state: error")
                    self.recent_error_var.set(f"Recent error: {payload}")
                    self.stitch_progress_var.set("Progress: failed")
                    self._set_status(str(payload))
                    self._record_mission_event("stitch_failed", payload)
                    self._request_stitching_plane_redraw(force=True)
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
        self._refresh_stitching_labels()

    def _mark_positions_available(self, positions: dict[str, int]) -> None:
        for axis in positions:
            if axis in self.position_available:
                self.position_available[axis] = True

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

    def _record_mission_event(self, kind: str, payload: object = None) -> None:
        message = mission_log_message_for_event(kind, payload)
        if message is not None:
            self._append_mission_log(message)

    def _append_mission_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"T+ {timestamp}  {message}"
        self.mission_log_lines.append(line)
        self.mission_log_lines = self.mission_log_lines[-80:]
        text_widget = getattr(self, "mission_log_text", None)
        if text_widget is None:
            return
        text_widget.configure(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, "\n".join(self.mission_log_lines))
        text_widget.configure(state="disabled")
        text_widget.see(tk.END)

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
