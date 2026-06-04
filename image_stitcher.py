"""Offline image stitching helpers for saved tile sessions."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from stitching_store import TileRecord

_COARSE_REGISTRATION_MAX_DIMENSION = 128
_INTERMEDIATE_REGISTRATION_MAX_DIMENSION = 512
_STREAMING_TILE_COUNT_SWITCH = 200
_STREAMING_MOSAIC_PIXEL_SWITCH = 60_000_000
_REGISTRATION_MIN_IMPROVEMENT_RATIO = 0.85


def stitch_tiles_by_stage_coordinates(
    frames: list[np.ndarray],
    tiles: list[TileRecord],
    *,
    pixels_per_pulse: float | None = None,
    x_pixels_per_pulse: float | None = None,
    y_pixels_per_pulse: float | None = None,
    use_overlap_registration: bool = False,
) -> np.ndarray:
    if len(frames) != len(tiles):
        raise ValueError("frames and tiles must have the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    x_scale, y_scale = _axis_scales(pixels_per_pulse, x_pixels_per_pulse, y_pixels_per_pulse)

    if use_overlap_registration:
        positions = refine_tile_positions_by_overlap(
            frames,
            tiles,
            x_pixels_per_pulse=x_scale,
            y_pixels_per_pulse=y_scale,
        )
    else:
        min_x = min(tile.x for tile in tiles)
        min_y = min(tile.y for tile in tiles)
        positions = {
            index: (
                int(round((tile.x - min_x) * x_scale)),
                int(round((tile.y - min_y) * y_scale)),
            )
            for index, tile in enumerate(tiles)
        }
    placements = []
    max_right = 0
    max_bottom = 0
    min_left = min(left for left, _top in positions.values())
    min_top = min(top for _left, top in positions.values())
    for index, (frame, tile) in enumerate(zip(frames, tiles)):
        if frame is None:
            raise ValueError(f"tile frame is missing: {tile.filename}")
        height, width = frame.shape[:2]
        left, top = positions[index]
        left -= min_left
        top -= min_top
        placements.append((frame, left, top))
        max_right = max(max_right, left + width)
        max_bottom = max(max_bottom, top + height)

    channels = 1 if frames[0].ndim == 2 else frames[0].shape[2]
    if channels == 1:
        accumulator = np.zeros((max_bottom, max_right), dtype=np.float32)
        weights = np.zeros((max_bottom, max_right), dtype=np.float32)
    else:
        accumulator = np.zeros((max_bottom, max_right, channels), dtype=np.float32)
        weights = np.zeros((max_bottom, max_right, 1), dtype=np.float32)

    for frame, left, top in placements:
        _accumulate_weighted_frame(accumulator, weights, frame, left, top, channels)

    weights = np.maximum(weights, 1.0)
    mosaic = accumulator / weights
    return np.clip(mosaic, 0, 255).astype(frames[0].dtype)


def _feather_weights(height: int, width: int) -> np.ndarray:
    """Favor tile interiors so overlap transitions do not form hard seams."""
    x_distance = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1)).astype(np.float32)
    y_distance = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1)).astype(np.float32)
    return np.outer(y_distance, x_distance)


class IncrementalMosaicBuilder:
    """Accumulate a coordinate-based mosaic tile by tile."""

    def __init__(
        self,
        tiles: list[TileRecord],
        frame_shape: tuple[int, ...],
        *,
        pixels_per_pulse: float | None = None,
        x_pixels_per_pulse: float | None = None,
        y_pixels_per_pulse: float | None = None,
    ) -> None:
        if not tiles:
            raise ValueError("at least one tile is required")
        self.tiles = tiles
        self.frame_shape = tuple(frame_shape)
        self.positions = coordinate_tile_positions(
            tiles,
            pixels_per_pulse=pixels_per_pulse,
            x_pixels_per_pulse=x_pixels_per_pulse,
            y_pixels_per_pulse=y_pixels_per_pulse,
        )
        self.mosaic_height, self.mosaic_width, self.channels = estimate_mosaic_shape(
            [self.frame_shape for _tile in tiles],
            tiles,
            pixels_per_pulse=pixels_per_pulse,
            x_pixels_per_pulse=x_pixels_per_pulse,
            y_pixels_per_pulse=y_pixels_per_pulse,
        )
        self._temp_dir = tempfile.TemporaryDirectory(prefix="stitch_incremental_")
        temp_path = Path(self._temp_dir.name)
        if self.channels == 1:
            self.accumulator = np.memmap(
                temp_path / "accumulator.dat",
                dtype=np.float32,
                mode="w+",
                shape=(self.mosaic_height, self.mosaic_width),
            )
            self.weights = np.memmap(
                temp_path / "weights.dat",
                dtype=np.float32,
                mode="w+",
                shape=(self.mosaic_height, self.mosaic_width),
            )
        else:
            self.accumulator = np.memmap(
                temp_path / "accumulator.dat",
                dtype=np.float32,
                mode="w+",
                shape=(self.mosaic_height, self.mosaic_width, self.channels),
            )
            self.weights = np.memmap(
                temp_path / "weights.dat",
                dtype=np.float32,
                mode="w+",
                shape=(self.mosaic_height, self.mosaic_width, 1),
            )
        self.accumulator[:] = 0
        self.weights[:] = 0
        self._dtype = np.dtype(np.uint8)
        self._closed = False

    def add_tile(self, index: int, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("incremental mosaic builder is closed")
        if index < 0 or index >= len(self.tiles):
            raise IndexError("tile index out of range")
        if frame.shape[:2] != self.frame_shape[:2]:
            raise ValueError("all incremental tiles must have the calibrated frame size")
        self._dtype = frame.dtype
        left, top = self.positions[index]
        _accumulate_weighted_frame(self.accumulator, self.weights, frame, left, top, self.channels)

    def write(self, output: Path | str) -> Path:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to write stitched mosaics") from exc

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(cv2.imwrite(str(output_path), self.to_array()))
        if not ok:
            raise RuntimeError(f"failed to write stitched mosaic: {output_path}")
        return output_path

    def write_with_boundaries(self, output: Path | str) -> Path:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to write stitched mosaics") from exc

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mosaic = draw_tile_boundaries(self.to_array(), self.positions, self.frame_shape)
        ok = bool(cv2.imwrite(str(output_path), mosaic))
        if not ok:
            raise RuntimeError(f"failed to write stitched mosaic boundaries: {output_path}")
        return output_path

    def to_array(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("incremental mosaic builder is closed")
        mosaic = self.accumulator / np.maximum(self.weights, 1.0)
        return np.clip(mosaic, 0, 255).astype(self._dtype)

    def close(self) -> None:
        if self._closed:
            return
        self.accumulator.flush()
        self.weights.flush()
        self._temp_dir.cleanup()
        self._closed = True

    def __enter__(self) -> "IncrementalMosaicBuilder":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def _accumulate_weighted_frame(accumulator, weights, frame: np.ndarray, left: int, top: int, channels: int) -> None:
    height, width = frame.shape[:2]
    view = accumulator[top : top + height, left : left + width]
    weight_view = weights[top : top + height, left : left + width]
    feather = _feather_weights(height, width)
    if channels != 1:
        feather = feather[..., None]
    view += frame.astype(np.float32) * feather
    weight_view += feather


def draw_tile_boundaries(
    mosaic: np.ndarray,
    positions: dict[int, tuple[int, int]],
    frame_shape: tuple[int, ...],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    debug = np.asarray(mosaic).copy()
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if debug.ndim == 2:
        boundary_value = max(color)
    else:
        boundary_value = np.asarray(color[: debug.shape[2]], dtype=debug.dtype)
    for left, top in positions.values():
        right = min(debug.shape[1] - 1, left + width - 1)
        bottom = min(debug.shape[0] - 1, top + height - 1)
        left = max(0, left)
        top = max(0, top)
        if right < left or bottom < top:
            continue
        debug[top : bottom + 1, left] = boundary_value
        debug[top : bottom + 1, right] = boundary_value
        debug[top, left : right + 1] = boundary_value
        debug[bottom, left : right + 1] = boundary_value
    return debug


def coordinate_tile_positions(
    tiles: list[TileRecord],
    *,
    pixels_per_pulse: float | None = None,
    x_pixels_per_pulse: float | None = None,
    y_pixels_per_pulse: float | None = None,
) -> dict[int, tuple[int, int]]:
    x_scale, y_scale = _axis_scales(pixels_per_pulse, x_pixels_per_pulse, y_pixels_per_pulse)
    min_x = min(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    raw_positions = {
        index: (
            int(round((tile.x - min_x) * x_scale)),
            int(round((tile.y - min_y) * y_scale)),
        )
        for index, tile in enumerate(tiles)
    }
    min_left = min(left for left, _top in raw_positions.values())
    min_top = min(top for _left, top in raw_positions.values())
    return {index: (left - min_left, top - min_top) for index, (left, top) in raw_positions.items()}


def estimate_mosaic_shape(
    frame_shapes: list[tuple[int, ...]],
    tiles: list[TileRecord],
    *,
    pixels_per_pulse: float | None = None,
    x_pixels_per_pulse: float | None = None,
    y_pixels_per_pulse: float | None = None,
) -> tuple[int, int, int]:
    if len(frame_shapes) != len(tiles):
        raise ValueError("frame_shapes and tiles must have the same length")
    positions = coordinate_tile_positions(
        tiles,
        pixels_per_pulse=pixels_per_pulse,
        x_pixels_per_pulse=x_pixels_per_pulse,
        y_pixels_per_pulse=y_pixels_per_pulse,
    )
    max_right = 0
    max_bottom = 0
    channels = 1 if len(frame_shapes[0]) == 2 else int(frame_shapes[0][2])
    for index, shape in enumerate(frame_shapes):
        height, width = int(shape[0]), int(shape[1])
        left, top = positions[index]
        max_right = max(max_right, left + width)
        max_bottom = max(max_bottom, top + height)
    return max_bottom, max_right, channels


def refine_tile_positions_by_overlap(
    frames: list[np.ndarray],
    tiles: list[TileRecord],
    *,
    pixels_per_pulse: float | None = None,
    x_pixels_per_pulse: float | None = None,
    y_pixels_per_pulse: float | None = None,
    max_correction: int = 60,
) -> dict[int, tuple[int, int]]:
    """Refine stage-derived tile positions by matching adjacent overlaps."""
    if len(frames) != len(tiles):
        raise ValueError("frames and tiles must have the same length")
    x_scale, y_scale = _axis_scales(pixels_per_pulse, x_pixels_per_pulse, y_pixels_per_pulse)
    if not frames:
        return {}

    min_x = min(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    expected = {
        index: (
            int(round((tile.x - min_x) * x_scale)),
            int(round((tile.y - min_y) * y_scale)),
        )
        for index, tile in enumerate(tiles)
    }
    by_grid = {(tile.row, tile.col): index for index, tile in enumerate(tiles)}
    positions: dict[int, tuple[int, int]] = {}

    for index, tile in enumerate(tiles):
        if index == 0:
            positions[index] = expected[index]
            continue

        left_index = by_grid.get((tile.row, tile.col - 1))
        up_index = by_grid.get((tile.row - 1, tile.col))
        if left_index is not None and left_index in positions:
            anchor = positions[left_index]
            expected_delta = (
                expected[index][0] - expected[left_index][0],
                expected[index][1] - expected[left_index][1],
            )
            delta = _estimate_neighbor_delta(
                frames[left_index],
                frames[index],
                expected_delta=expected_delta,
                max_correction=max_correction,
            )
            positions[index] = (anchor[0] + delta[0], anchor[1] + delta[1])
        elif up_index is not None and up_index in positions:
            anchor = positions[up_index]
            expected_delta = (
                expected[index][0] - expected[up_index][0],
                expected[index][1] - expected[up_index][1],
            )
            delta = _estimate_neighbor_delta(
                frames[up_index],
                frames[index],
                expected_delta=expected_delta,
                max_correction=max_correction,
            )
            positions[index] = (anchor[0] + delta[0], anchor[1] + delta[1])
        else:
            positions[index] = expected[index]
    return positions


def _estimate_neighbor_delta(
    anchor_frame: np.ndarray,
    moving_frame: np.ndarray,
    *,
    expected_delta: tuple[int, int],
    max_correction: int,
) -> tuple[int, int]:
    anchor_gray = _as_gray_float(anchor_frame)
    moving_gray = _as_gray_float(moving_frame)
    max_dimension = max(anchor_gray.shape + moving_gray.shape)
    coarse_scale = min(1.0, _COARSE_REGISTRATION_MAX_DIMENSION / max_dimension)
    if coarse_scale >= 1.0:
        delta, score = _search_neighbor_delta(
            anchor_gray,
            moving_gray,
            center=expected_delta,
            radius=max_correction,
            allowed_center=expected_delta,
            allowed_radius=max_correction,
        )
        return _accept_neighbor_delta(anchor_gray, moving_gray, delta, score, expected_delta)

    coarse_anchor = _resize_gray(anchor_gray, coarse_scale)
    coarse_moving = _resize_gray(moving_gray, coarse_scale)
    coarse_expected = _scale_delta(expected_delta, coarse_scale)
    coarse_delta, _coarse_score = _search_neighbor_delta(
        coarse_anchor,
        coarse_moving,
        center=coarse_expected,
        radius=max(1, int(np.ceil(max_correction * coarse_scale))),
    )
    estimate = _unscale_delta(coarse_delta, coarse_scale)

    intermediate_scale = min(1.0, _INTERMEDIATE_REGISTRATION_MAX_DIMENSION / max_dimension)
    if intermediate_scale > coarse_scale and intermediate_scale < 1.0:
        intermediate_anchor = _resize_gray(anchor_gray, intermediate_scale)
        intermediate_moving = _resize_gray(moving_gray, intermediate_scale)
        intermediate_center = _scale_delta(estimate, intermediate_scale)
        intermediate_delta, _intermediate_score = _search_neighbor_delta(
            intermediate_anchor,
            intermediate_moving,
            center=intermediate_center,
            radius=max(2, int(np.ceil(intermediate_scale / coarse_scale)) + 1),
        )
        estimate = _unscale_delta(intermediate_delta, intermediate_scale)

    delta, score = _search_neighbor_delta(
        anchor_gray,
        moving_gray,
        center=estimate,
        radius=2,
        allowed_center=expected_delta,
        allowed_radius=max_correction,
    )
    return _accept_neighbor_delta(anchor_gray, moving_gray, delta, score, expected_delta)


def _search_neighbor_delta(
    anchor_gray: np.ndarray,
    moving_gray: np.ndarray,
    *,
    center: tuple[int, int],
    radius: int,
    allowed_center: tuple[int, int] | None = None,
    allowed_radius: int | None = None,
) -> tuple[tuple[int, int], float | None]:
    best_delta = center
    best_score: float | None = None
    cx, cy = center
    x_candidates = range(cx - radius, cx + radius + 1)
    y_candidates = range(cy - radius, cy + radius + 1)
    for dy in y_candidates:
        for dx in x_candidates:
            if allowed_center is not None and allowed_radius is not None:
                if abs(dx - allowed_center[0]) > allowed_radius or abs(dy - allowed_center[1]) > allowed_radius:
                    continue
            score = _overlap_mse(anchor_gray, moving_gray, dx, dy)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_delta = (dx, dy)
    return best_delta, best_score


def _accept_neighbor_delta(
    anchor_gray: np.ndarray,
    moving_gray: np.ndarray,
    candidate_delta: tuple[int, int],
    candidate_score: float | None,
    expected_delta: tuple[int, int],
) -> tuple[int, int]:
    if candidate_delta == expected_delta:
        return expected_delta
    if candidate_score is None:
        return expected_delta
    expected_score = _overlap_mse(anchor_gray, moving_gray, expected_delta[0], expected_delta[1])
    if expected_score is None:
        return expected_delta
    if candidate_score <= expected_score * _REGISTRATION_MIN_IMPROVEMENT_RATIO:
        return candidate_delta
    return expected_delta


def _axis_scales(
    pixels_per_pulse: float | None,
    x_pixels_per_pulse: float | None,
    y_pixels_per_pulse: float | None,
) -> tuple[float, float]:
    default_scale = 1.0 if pixels_per_pulse is None else float(pixels_per_pulse)
    x_scale = default_scale if x_pixels_per_pulse is None else float(x_pixels_per_pulse)
    y_scale = default_scale if y_pixels_per_pulse is None else float(y_pixels_per_pulse)
    if x_scale == 0.0 or y_scale == 0.0:
        raise ValueError("pixel-per-pulse calibration must be non-zero")
    return x_scale, y_scale


def _as_gray_float(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype(np.float32)


def _resize_gray(frame: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _scale_delta(delta: tuple[int, int], scale: float) -> tuple[int, int]:
    return int(round(delta[0] * scale)), int(round(delta[1] * scale))


def _unscale_delta(delta: tuple[int, int], scale: float) -> tuple[int, int]:
    return int(round(delta[0] / scale)), int(round(delta[1] / scale))


def _overlap_mse(anchor: np.ndarray, moving: np.ndarray, dx: int, dy: int) -> float | None:
    anchor_h, anchor_w = anchor.shape[:2]
    moving_h, moving_w = moving.shape[:2]
    left = max(0, dx)
    top = max(0, dy)
    right = min(anchor_w, dx + moving_w)
    bottom = min(anchor_h, dy + moving_h)
    if right - left < 8 or bottom - top < 8:
        return None

    moving_left = left - dx
    moving_top = top - dy
    anchor_roi = anchor[top:bottom, left:right]
    moving_roi = moving[moving_top : moving_top + (bottom - top), moving_left : moving_left + (right - left)]
    if float(anchor_roi.std()) < 1.0 or float(moving_roi.std()) < 1.0:
        return None
    anchor_norm = anchor_roi - float(anchor_roi.mean())
    moving_norm = moving_roi - float(moving_roi.mean())
    return float(np.mean((anchor_norm - moving_norm) ** 2))


def stitch_session_by_metadata(
    session_path: Path | str,
    *,
    pixels_per_pulse: float | None = None,
    output_name: str = "stitched_mosaic.png",
    use_overlap_registration: bool = True,
    stream_after_tile_count: int = _STREAMING_TILE_COUNT_SWITCH,
    stream_after_mosaic_pixels: int = _STREAMING_MOSAIC_PIXEL_SWITCH,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to stitch saved sessions") from exc

    root = Path(session_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    calibration = metadata.get("calibration", {})
    x_scale = calibration.get("x_pixels_per_pulse") if pixels_per_pulse is None else None
    y_scale = calibration.get("y_pixels_per_pulse") if pixels_per_pulse is None else None
    tiles = [TileRecord(**item) for item in metadata.get("tiles", [])]
    if not tiles:
        raise RuntimeError("metadata contains no stitching tiles")

    frame_shapes = []
    for tile in tiles:
        frame = cv2.imread(str(root / tile.filename), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read tile image: {tile.filename}")
        frame_shapes.append(frame.shape)
    mosaic_height, mosaic_width, _channels = estimate_mosaic_shape(
        frame_shapes,
        tiles,
        pixels_per_pulse=pixels_per_pulse,
        x_pixels_per_pulse=x_scale,
        y_pixels_per_pulse=y_scale,
    )
    mosaic_pixels = mosaic_height * mosaic_width
    should_stream = len(tiles) >= stream_after_tile_count or mosaic_pixels >= stream_after_mosaic_pixels
    if should_stream:
        return stitch_session_by_metadata_streaming(
            root,
            tiles,
            frame_shapes,
            pixels_per_pulse=pixels_per_pulse,
            x_pixels_per_pulse=x_scale,
            y_pixels_per_pulse=y_scale,
            output_name=output_name,
        )

    frames = []
    for tile in tiles:
        frame = cv2.imread(str(root / tile.filename), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read tile image: {tile.filename}")
        frames.append(frame)
    mosaic = stitch_tiles_by_stage_coordinates(
        frames,
        tiles,
        pixels_per_pulse=pixels_per_pulse,
        x_pixels_per_pulse=x_scale,
        y_pixels_per_pulse=y_scale,
        use_overlap_registration=use_overlap_registration,
    )
    output = root / output_name
    ok = bool(cv2.imwrite(str(output), mosaic))
    if not ok:
        raise RuntimeError(f"failed to write stitched mosaic: {output}")
    return output


def stitch_session_by_metadata_streaming(
    root: Path,
    tiles: list[TileRecord],
    frame_shapes: list[tuple[int, ...]],
    *,
    pixels_per_pulse: float | None,
    x_pixels_per_pulse: float | None,
    y_pixels_per_pulse: float | None,
    output_name: str,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to stitch saved sessions") from exc

    mosaic_height, mosaic_width, channels = estimate_mosaic_shape(
        frame_shapes,
        tiles,
        pixels_per_pulse=pixels_per_pulse,
        x_pixels_per_pulse=x_pixels_per_pulse,
        y_pixels_per_pulse=y_pixels_per_pulse,
    )
    positions = coordinate_tile_positions(
        tiles,
        pixels_per_pulse=pixels_per_pulse,
        x_pixels_per_pulse=x_pixels_per_pulse,
        y_pixels_per_pulse=y_pixels_per_pulse,
    )
    output = root / output_name
    with tempfile.TemporaryDirectory(prefix="stitch_accum_") as temp_dir:
        temp_path = Path(temp_dir)
        if channels == 1:
            accumulator = np.memmap(temp_path / "accumulator.dat", dtype=np.float32, mode="w+", shape=(mosaic_height, mosaic_width))
            weights = np.memmap(temp_path / "weights.dat", dtype=np.float32, mode="w+", shape=(mosaic_height, mosaic_width))
        else:
            accumulator = np.memmap(
                temp_path / "accumulator.dat",
                dtype=np.float32,
                mode="w+",
                shape=(mosaic_height, mosaic_width, channels),
            )
            weights = np.memmap(temp_path / "weights.dat", dtype=np.float32, mode="w+", shape=(mosaic_height, mosaic_width, 1))
        accumulator[:] = 0
        weights[:] = 0

        for index, tile in enumerate(tiles):
            read_mode = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
            frame = cv2.imread(str(root / tile.filename), read_mode)
            if frame is None:
                raise RuntimeError(f"failed to read tile image: {tile.filename}")
            height, width = frame.shape[:2]
            left, top = positions[index]
            view = accumulator[top : top + height, left : left + width]
            weight_view = weights[top : top + height, left : left + width]
            feather = _feather_weights(height, width)
            if channels != 1:
                feather = feather[..., None]
            view += frame.astype(np.float32) * feather
            weight_view += feather

        weights[:] = np.maximum(weights, 1.0)
        if channels == 1:
            mosaic = np.memmap(temp_path / "mosaic.dat", dtype=np.uint8, mode="w+", shape=(mosaic_height, mosaic_width))
        else:
            mosaic = np.memmap(temp_path / "mosaic.dat", dtype=np.uint8, mode="w+", shape=(mosaic_height, mosaic_width, channels))
        rows_per_chunk = max(1, min(512, 64_000_000 // max(1, mosaic_width * max(1, channels))))
        for start in range(0, mosaic_height, rows_per_chunk):
            stop = min(mosaic_height, start + rows_per_chunk)
            mosaic[start:stop] = np.clip(accumulator[start:stop] / weights[start:stop], 0, 255).astype(np.uint8)
        accumulator.flush()
        weights.flush()
        mosaic.flush()
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
