"""Normalized view model for the image-stitching plane preview."""

from __future__ import annotations

from dataclasses import dataclass

from sample_plane import SamplePlanePoint
from scan_plan import TilePoint


@dataclass(frozen=True)
class ViewPoint:
    label: str
    nx: float
    ny: float
    z: int
    state: str = "normal"


@dataclass(frozen=True)
class StitchingViewModel:
    corners: list[ViewPoint]
    tiles: list[ViewPoint]


def build_stitching_view_model(
    corners: list[SamplePlanePoint],
    tiles: list[TilePoint],
    *,
    current_tile_index: int | None = None,
    captured_tile_count: int | None = None,
) -> StitchingViewModel:
    xs = [point.x for point in corners] + [tile.x for tile in tiles]
    ys = [point.y for point in corners] + [tile.y for tile in tiles]
    if not xs or not ys:
        return StitchingViewModel(corners=[], tiles=[])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1, max_x - min_x)
    span_y = max(1, max_y - min_y)

    corner_points = [
        ViewPoint(label=corner.label, nx=(corner.x - min_x) / span_x, ny=(corner.y - min_y) / span_y, z=corner.z)
        for corner in corners
    ]
    tile_points = []
    for index, tile in enumerate(tiles):
        completed_count = current_tile_index if captured_tile_count is None else captured_tile_count
        if completed_count is not None and index < completed_count:
            state = "done"
        elif current_tile_index is not None and index == current_tile_index:
            state = "current"
        else:
            state = "pending"
        tile_points.append(
            ViewPoint(
                label=f"{tile.row},{tile.col}",
                nx=(tile.x - min_x) / span_x,
                ny=(tile.y - min_y) / span_y,
                z=tile.z,
                state=state,
            )
        )
    return StitchingViewModel(corners=corner_points, tiles=tile_points)
