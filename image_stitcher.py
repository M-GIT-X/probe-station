"""Offline image stitching helpers for saved tile sessions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stitching_store import TileRecord


def stitch_tiles_by_stage_coordinates(
    frames: list[np.ndarray],
    tiles: list[TileRecord],
    *,
    pixels_per_pulse: float,
) -> np.ndarray:
    if len(frames) != len(tiles):
        raise ValueError("frames and tiles must have the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    if pixels_per_pulse <= 0:
        raise ValueError("pixels_per_pulse must be positive")

    min_x = min(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    placements = []
    max_right = 0
    max_bottom = 0
    for frame, tile in zip(frames, tiles):
        if frame is None:
            raise ValueError(f"tile frame is missing: {tile.filename}")
        height, width = frame.shape[:2]
        left = int(round((tile.x - min_x) * pixels_per_pulse))
        top = int(round((tile.y - min_y) * pixels_per_pulse))
        placements.append((frame, left, top))
        max_right = max(max_right, left + width)
        max_bottom = max(max_bottom, top + height)

    channels = 1 if frames[0].ndim == 2 else frames[0].shape[2]
    if channels == 1:
        accumulator = np.zeros((max_bottom, max_right), dtype=np.float64)
        weights = np.zeros((max_bottom, max_right), dtype=np.float64)
    else:
        accumulator = np.zeros((max_bottom, max_right, channels), dtype=np.float64)
        weights = np.zeros((max_bottom, max_right, 1), dtype=np.float64)

    for frame, left, top in placements:
        height, width = frame.shape[:2]
        view = accumulator[top : top + height, left : left + width]
        weight_view = weights[top : top + height, left : left + width]
        view += frame.astype(np.float64)
        weight_view += 1.0

    weights = np.maximum(weights, 1.0)
    mosaic = accumulator / weights
    return np.clip(mosaic, 0, 255).astype(frames[0].dtype)


def stitch_session_by_metadata(
    session_path: Path | str,
    *,
    pixels_per_pulse: float = 1.0,
    output_name: str = "stitched_mosaic.png",
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to stitch saved sessions") from exc

    root = Path(session_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    tiles = [TileRecord(**item) for item in metadata.get("tiles", [])]
    frames = []
    for tile in tiles:
        frame = cv2.imread(str(root / tile.filename), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read tile image: {tile.filename}")
        frames.append(frame)
    mosaic = stitch_tiles_by_stage_coordinates(frames, tiles, pixels_per_pulse=pixels_per_pulse)
    output = root / output_name
    ok = bool(cv2.imwrite(str(output), mosaic))
    if not ok:
        raise RuntimeError(f"failed to write stitched mosaic: {output}")
    return output


def try_opencv_stitcher(frames: list[np.ndarray]) -> np.ndarray | None:
    """Try OpenCV's panorama stitcher; return None when it cannot solve the set."""
    try:
        import cv2
    except ImportError:
        return None
    if len(frames) < 2 or not hasattr(cv2, "Stitcher_create"):
        return None
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, result = stitcher.stitch(frames)
    if status != cv2.Stitcher_OK:
        return None
    return result
