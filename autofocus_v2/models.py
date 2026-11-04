"""Shared measurement types. Host receive times are not exposure timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.x, self.y, self.width, self.height)):
            raise ValueError("ROI coordinates must be integers")
        if self.x < 0 or self.y < 0 or self.width < 16 or self.height < 16:
            raise ValueError("ROI must be nonnegative and at least 16 × 16 pixels")

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) not in (2, 3) or self.x + self.width > shape[1] or self.y + self.height > shape[0]:
            raise ValueError("ROI extends beyond the image")


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    read_started: float
    received_at: float
    image: np.ndarray


@dataclass(frozen=True)
class TileScore:
    # Tuple permits subtiles smaller than the minimum user-selectable ROI.
    bounds: tuple[int, int, int, int]
    gradient: float
    laplacian: float
    mean_brightness: float
    saturation_fraction: float
    dark_fraction: float
    valid: bool
    reasons: tuple[str, ...] = ()
    brightness_std: float = 0.0


@dataclass(frozen=True)
class FrameScore:
    roi: ROI
    score: float
    tiles: tuple[TileScore, ...]
    valid: bool
    reasons: tuple[str, ...] = ()
    method: str = "gradient_v1"


@dataclass(frozen=True)
class FrameRecord:
    sequence: int
    read_started: float
    received_at: float
    image_path: str
    metrics: FrameScore


@dataclass(frozen=True)
class Measurement:
    sample_id: str
    z: float
    z_source: str
    frames: tuple[FrameRecord, ...]
    notes: str = ""


@dataclass(frozen=True)
class BurstSummary:
    z: float
    score: float
    spread: float
    tile_scores: tuple[float, ...]
    frame_count: int
    valid: bool
    reasons: tuple[str, ...] = ()
    tile_spreads: tuple[float, ...] = ()
    brightness_spread: float = 0.0
    valid_frame_count: int = 0
    tile_valid: tuple[bool, ...] = ()


@dataclass(frozen=True)
class FocusAnalysis:
    status: str
    message: str
    points: tuple[BurstSummary, ...]
    best_z: float | None = None
    peak_interval: tuple[float, float] | None = None
    warnings: tuple[str, ...] = ()
