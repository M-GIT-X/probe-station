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
