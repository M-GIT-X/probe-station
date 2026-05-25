"""Safe in-bound camera scale calibration for automatic image stitching."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sample_plane import SamplePlane
from scan_plan import ScanBounds, TilePoint, point_in_polygon


@dataclass(frozen=True)
class CalibrationTrialPlan:
    reference: TilePoint
    x_trial: TilePoint
    y_trial: TilePoint
    x_step_pulses: int
    y_step_pulses: int


@dataclass(frozen=True)
class StitchingCalibration:
    x_pixels_per_pulse: float
    y_pixels_per_pulse: float
    x_step_pulses: int
    y_step_pulses: int
    x_frame_shift: tuple[float, float]
    y_frame_shift: tuple[float, float]
    frame_width: int
    frame_height: int
    overlap_percent: float
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "StitchingCalibration":
        return cls(
            x_pixels_per_pulse=float(values["x_pixels_per_pulse"]),
            y_pixels_per_pulse=float(values["y_pixels_per_pulse"]),
            x_step_pulses=int(values["x_step_pulses"]),
            y_step_pulses=int(values["y_step_pulses"]),
            x_frame_shift=tuple(float(value) for value in values["x_frame_shift"]),  # type: ignore[arg-type]
            y_frame_shift=tuple(float(value) for value in values["y_frame_shift"]),  # type: ignore[arg-type]
            frame_width=int(values["frame_width"]),
            frame_height=int(values["frame_height"]),
            overlap_percent=float(values["overlap_percent"]),
            confidence=float(values["confidence"]),
        )


def build_in_bounds_trial_plan(
    bounds: ScanBounds,
    plane: SamplePlane,
    *,
    boundary_points: list[tuple[int, int]] | None = None,
    preferred_step_pulses: int | None = None,
) -> CalibrationTrialPlan:
    span_x = bounds.max_x - bounds.min_x
    span_y = bounds.max_y - bounds.min_y
    if span_x < 4 or span_y < 4:
        raise ValueError("four-corner scan bounds are too small for automatic calibration")
    x_step = _trial_step(span_x, preferred_step_pulses)
    y_step = _trial_step(span_y, preferred_step_pulses)
    center_x = (bounds.min_x + bounds.max_x) // 2
    center_y = (bounds.min_y + bounds.max_y) // 2
    if boundary_points:
        center_x = int(round(sum(point[0] for point in boundary_points) / len(boundary_points)))
        center_y = int(round(sum(point[1] for point in boundary_points) / len(boundary_points)))
        x_step = _fit_trial_step(center_x, center_y, x_step, 0, boundary_points)
        y_step = _fit_trial_step(center_x, center_y, 0, y_step, boundary_points)
    reference = TilePoint(row=-1, col=-1, x=center_x, y=center_y, z=plane.z_at(center_x, center_y))
    x_trial = TilePoint(row=-1, col=-1, x=center_x + x_step, y=center_y, z=plane.z_at(center_x + x_step, center_y))
    y_trial = TilePoint(row=-1, col=-1, x=center_x, y=center_y + y_step, z=plane.z_at(center_x, center_y + y_step))
    return CalibrationTrialPlan(reference, x_trial, y_trial, x_step, y_step)


def calibration_from_shifts(
    *,
    x_step_pulses: int,
    y_step_pulses: int,
    x_frame_shift: tuple[float, float],
    y_frame_shift: tuple[float, float],
    frame_width: int,
    frame_height: int,
    overlap_percent: float,
    confidence: float,
) -> StitchingCalibration:
    x_movement_pixels = -float(x_frame_shift[0])
    y_movement_pixels = -float(y_frame_shift[1])
    if abs(x_movement_pixels) < 2.0 or abs(y_movement_pixels) < 2.0:
        raise ValueError("calibration image shift is too small to measure reliably")
    if x_step_pulses <= 0 or y_step_pulses <= 0:
        raise ValueError("calibration step pulses must be positive")
    return StitchingCalibration(
        x_pixels_per_pulse=x_movement_pixels / x_step_pulses,
        y_pixels_per_pulse=y_movement_pixels / y_step_pulses,
        x_step_pulses=x_step_pulses,
        y_step_pulses=y_step_pulses,
        x_frame_shift=x_frame_shift,
        y_frame_shift=y_frame_shift,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        overlap_percent=float(overlap_percent),
        confidence=float(confidence),
    )


def estimate_calibration_from_frames(
    reference_frame,
    x_trial_frame,
    y_trial_frame,
    trial: CalibrationTrialPlan,
    overlap_percent: float,
) -> StitchingCalibration:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for automatic stitching calibration") from exc

    reference = _gray_float(reference_frame)
    x_frame = _gray_float(x_trial_frame)
    y_frame = _gray_float(y_trial_frame)
    x_shift, x_response = cv2.phaseCorrelate(reference, x_frame)
    y_shift, y_response = cv2.phaseCorrelate(reference, y_frame)
    confidence = float(min(x_response, y_response))
    if confidence < 0.05:
        raise ValueError("automatic stitching calibration confidence is too low")
    height, width = reference.shape[:2]
    return calibration_from_shifts(
        x_step_pulses=trial.x_step_pulses,
        y_step_pulses=trial.y_step_pulses,
        x_frame_shift=(float(x_shift[0]), float(x_shift[1])),
        y_frame_shift=(float(y_shift[0]), float(y_shift[1])),
        frame_width=width,
        frame_height=height,
        overlap_percent=overlap_percent,
        confidence=confidence,
    )


def _trial_step(span: int, preferred_step_pulses: int | None) -> int:
    if preferred_step_pulses is not None:
        return max(1, min(span // 4, int(preferred_step_pulses)))
    return max(1, min(span // 4, 20))


def _fit_trial_step(
    center_x: int,
    center_y: int,
    dx: int,
    dy: int,
    boundary_points: list[tuple[int, int]],
) -> int:
    step = max(abs(dx), abs(dy))
    while step >= 1:
        x = center_x + (step if dx else 0)
        y = center_y + (step if dy else 0)
        if point_in_polygon(x, y, boundary_points):
            return step
        step //= 2
    raise ValueError("four-corner boundary is too small for in-bound calibration movement")


def _gray_float(frame):
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype("float32")
