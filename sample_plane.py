"""Sample plane fitting for image stitching and future probe clearance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
