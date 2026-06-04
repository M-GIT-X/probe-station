"""Sample plane fitting for image stitching and future probe clearance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SamplePlanePoint:
    label: str
    x: int
    y: int
    z: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class SamplePlane:
    a: float
    b: float
    c: float
    max_abs_residual: float
    rms_residual: float

    def z_at(self, x: float, y: float) -> int:
        return int(round(self.a * float(x) + self.b * float(y) + self.c))

    def to_dict(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "max_abs_residual": self.max_abs_residual,
            "rms_residual": self.rms_residual,
        }


@dataclass(frozen=True)
class PlaneConsistencyReport:
    plane: SamplePlane
    accepted: bool
    residuals: dict[str, float]
    confirmation_residuals: dict[str, float]
    max_confirmation_residual: float
    suspicious_label: str | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "residuals": self.residuals,
            "confirmation_residuals": self.confirmation_residuals,
            "max_confirmation_residual": self.max_confirmation_residual,
            "suspicious_label": self.suspicious_label,
            "message": self.message,
        }


@dataclass(frozen=True)
class RobustPlaneFitResult:
    plane: SamplePlane
    inliers: list[SamplePlanePoint]
    outliers: list[SamplePlanePoint]
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "inliers": [point.to_dict() for point in self.inliers],
            "outliers": [point.to_dict() for point in self.outliers],
            "message": self.message,
            "plane": self.plane.to_dict(),
        }


def fit_sample_plane(points: Iterable[SamplePlanePoint]) -> SamplePlane:
    measured = list(points)
    if len(measured) < 3:
        raise ValueError("at least three points are required to fit a sample plane")

    rows = np.asarray([[point.x, point.y, 1.0] for point in measured], dtype=np.float64)
    z_values = np.asarray([point.z for point in measured], dtype=np.float64)
    rank = np.linalg.matrix_rank(rows)
    if rank < 3:
        raise ValueError("sample plane points must not be collinear")

    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(rows, z_values, rcond=None)
    fitted = rows @ coefficients
    residuals = z_values - fitted
    max_abs = float(np.max(np.abs(residuals)))
    rms = float(np.sqrt(np.mean(residuals**2)))
    return SamplePlane(
        a=float(coefficients[0]),
        b=float(coefficients[1]),
        c=float(coefficients[2]),
        max_abs_residual=max_abs,
        rms_residual=rms,
    )


def fit_sample_plane_robust(
    points: Iterable[SamplePlanePoint],
    *,
    max_abs_residual: float = 30.0,
    min_inliers: int = 6,
) -> RobustPlaneFitResult:
    remaining = list(points)
    if len(remaining) < 3:
        raise ValueError("at least three points are required to fit a sample plane")
    min_inliers = max(3, min(int(min_inliers), len(remaining)))
    outliers: list[SamplePlanePoint] = []
    while True:
        plane = fit_sample_plane(remaining)
        residuals = [(point, abs(float(point.z - plane.z_at(point.x, point.y)))) for point in remaining]
        worst_point, worst_residual = max(residuals, key=lambda item: item[1])
        if worst_residual <= max_abs_residual or len(remaining) <= min_inliers:
            break
        outliers.append(worst_point)
        remaining = [point for point in remaining if point.label != worst_point.label]

    plane = fit_sample_plane(remaining)
    accepted = plane.max_abs_residual <= max_abs_residual
    message = (
        f"robust plane accepted: {len(remaining)} inliers, {len(outliers)} outliers, "
        f"max residual {plane.max_abs_residual:.1f} pulses"
        if accepted
        else (
            f"robust plane warning: {len(remaining)} inliers, {len(outliers)} outliers, "
            f"max residual {plane.max_abs_residual:.1f} pulses"
        )
    )
    return RobustPlaneFitResult(plane=plane, inliers=remaining, outliers=outliers, message=message)


def generate_plane_probe_points(
    boundary_points: Iterable[SamplePlanePoint],
    seed_plane: SamplePlane,
    *,
    grid_size: int = 3,
) -> list[SamplePlanePoint]:
    measured = list(boundary_points)
    if len(measured) < 3:
        raise ValueError("at least three boundary points are required to generate probe points")
    grid_size = max(2, int(grid_size))
    min_x = min(point.x for point in measured)
    max_x = max(point.x for point in measured)
    min_y = min(point.y for point in measured)
    max_y = max(point.y for point in measured)
    polygon = boundary_polygon_from_points(measured)
    x_values = _evenly_spaced_ints(min_x, max_x, grid_size)
    y_values = _evenly_spaced_ints(min_y, max_y, grid_size)
    probes: list[SamplePlanePoint] = []
    for row, y in enumerate(y_values):
        for col, x in enumerate(x_values):
            if _point_in_polygon(x, y, polygon):
                probes.append(SamplePlanePoint(f"p{row + 1}{col + 1}", x, y, seed_plane.z_at(x, y)))
    if len(probes) < 3:
        raise ValueError("boundary is too small or irregular for automatic autofocus probe grid")
    return probes


def evaluate_plane_consistency(
    points: Iterable[SamplePlanePoint],
    *,
    max_confirmation_residual: float = 30.0,
    outlier_gap_ratio: float = 1.35,
) -> PlaneConsistencyReport:
    measured = list(points)
    if len(measured) != 4:
        raise ValueError("exactly four points are required to confirm a stitching plane")

    plane = fit_sample_plane(measured)
    residuals = {point.label: float(point.z - plane.z_at(point.x, point.y)) for point in measured}
    confirmation_residuals: dict[str, float] = {}
    for held_out in measured:
        support_points = [point for point in measured if point.label != held_out.label]
        confirmation_plane = fit_sample_plane(support_points)
        confirmation_residuals[held_out.label] = abs(float(held_out.z - confirmation_plane.z_at(held_out.x, held_out.y)))

    ranked = sorted(confirmation_residuals.items(), key=lambda item: item[1], reverse=True)
    worst_label, worst_residual = ranked[0]
    second_residual = ranked[1][1]
    accepted = worst_residual <= max_confirmation_residual
    suspicious_label = None
    if not accepted and worst_residual >= max(second_residual * outlier_gap_ratio, max_confirmation_residual):
        suspicious_label = worst_label

    if accepted:
        message = f"plane accepted: max confirmation residual {worst_residual:.1f} pulses"
    elif suspicious_label:
        message = (
            f"plane rejected: corner {suspicious_label} has confirmation residual "
            f"{worst_residual:.1f} pulses"
        )
    else:
        message = f"plane rejected: max confirmation residual {worst_residual:.1f} pulses"

    return PlaneConsistencyReport(
        plane=plane,
        accepted=accepted,
        residuals=residuals,
        confirmation_residuals=confirmation_residuals,
        max_confirmation_residual=worst_residual,
        suspicious_label=suspicious_label,
        message=message,
    )


def bounds_from_plane_points(points: Iterable[SamplePlanePoint]):
    from scan_plan import ScanBounds

    measured = list(points)
    if not measured:
        raise ValueError("at least one point is required to compute bounds")
    return ScanBounds(
        min_x=min(point.x for point in measured),
        max_x=max(point.x for point in measured),
        min_y=min(point.y for point in measured),
        max_y=max(point.y for point in measured),
    )


def boundary_polygon_from_points(points: Iterable[SamplePlanePoint]) -> list[tuple[int, int]]:
    measured = list(points)
    if len(measured) < 3:
        raise ValueError("at least three points are required to make a boundary")
    center_x = sum(point.x for point in measured) / len(measured)
    center_y = sum(point.y for point in measured) / len(measured)
    ordered = sorted(measured, key=lambda point: math.atan2(point.y - center_y, point.x - center_x))
    return [(point.x, point.y) for point in ordered]


def _evenly_spaced_ints(start: int, stop: int, count: int) -> list[int]:
    if count == 1:
        return [int(round((start + stop) / 2))]
    span = stop - start
    return [int(round(start + span * index / (count - 1))) for index in range(count)]


def _point_in_polygon(x: int, y: int, points: list[tuple[int, int]]) -> bool:
    if len(points) < 3:
        return False
    inside = False
    previous_x, previous_y = points[-1]
    for current_x, current_y in points:
        if _point_on_segment(x, y, previous_x, previous_y, current_x, current_y):
            return True
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_segment(x: int, y: int, ax: int, ay: int, bx: int, by: int) -> bool:
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if cross != 0:
        return False
    return min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by)
