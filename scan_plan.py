"""Scan path generation for four-corner image stitching."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sample_plane import SamplePlane


@dataclass(frozen=True)
class ScanBounds:
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TilePoint:
    row: int
    col: int
    x: int
    y: int
    z: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class OverlapScanPlan:
    rows: int
    cols: int
    x_step_pulses: int
    y_step_pulses: int
    overlap_percent: float
    tiles: list[TilePoint]


def generate_stitching_grid(bounds: ScanBounds, rows: int, cols: int, plane: SamplePlane) -> list[TilePoint]:
    rows = int(rows)
    cols = int(cols)
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be at least 1")
    if bounds.max_x < bounds.min_x or bounds.max_y < bounds.min_y:
        raise ValueError("scan bounds are invalid")

    y_values = _evenly_spaced_ints(bounds.min_y, bounds.max_y, rows)
    x_values = _evenly_spaced_ints(bounds.min_x, bounds.max_x, cols)

    tiles: list[TilePoint] = []
    for row, y in enumerate(y_values):
        col_iter = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_iter:
            x = x_values[col]
            tiles.append(TilePoint(row=row, col=col, x=x, y=y, z=plane.z_at(x, y)))
    return tiles


def generate_overlap_scan_plan(
    bounds: ScanBounds,
    plane: SamplePlane,
    *,
    frame_width: int,
    frame_height: int,
    x_pixels_per_pulse: float,
    y_pixels_per_pulse: float,
    overlap_percent: float,
    boundary_points: list[tuple[int, int]] | None = None,
) -> OverlapScanPlan:
    if not 0.0 <= overlap_percent < 90.0:
        raise ValueError("overlap percent must be between 0 and 90")
    if abs(x_pixels_per_pulse) <= 0.0 or abs(y_pixels_per_pulse) <= 0.0:
        raise ValueError("pixel-per-pulse calibration must be non-zero")
    coverage_ratio = 1.0 - overlap_percent / 100.0
    max_x_step = max(1.0, frame_width * coverage_ratio / abs(x_pixels_per_pulse))
    max_y_step = max(1.0, frame_height * coverage_ratio / abs(y_pixels_per_pulse))
    span_x = bounds.max_x - bounds.min_x
    span_y = bounds.max_y - bounds.min_y
    cols = 1 if span_x == 0 else int(__import__("math").ceil(span_x / max_x_step)) + 1
    rows = 1 if span_y == 0 else int(__import__("math").ceil(span_y / max_y_step)) + 1
    x_step = 0 if cols == 1 else int(round(span_x / (cols - 1)))
    y_step = 0 if rows == 1 else int(round(span_y / (rows - 1)))
    tiles = generate_stitching_grid(bounds, rows, cols, plane)
    if boundary_points:
        tiles = [tile for tile in tiles if point_in_polygon(tile.x, tile.y, boundary_points)]
    return OverlapScanPlan(rows, cols, x_step, y_step, overlap_percent, tiles)


def _evenly_spaced_ints(start: int, stop: int, count: int) -> list[int]:
    if count == 1:
        return [int(round((start + stop) / 2))]
    span = stop - start
    return [int(round(start + span * index / (count - 1))) for index in range(count)]


def point_in_polygon(x: int, y: int, points: list[tuple[int, int]]) -> bool:
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
