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


def _evenly_spaced_ints(start: int, stop: int, count: int) -> list[int]:
    if count == 1:
        return [int(round((start + stop) / 2))]
    span = stop - start
    return [int(round(start + span * index / (count - 1))) for index in range(count)]
