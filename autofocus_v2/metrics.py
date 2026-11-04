"""Unwarped, fixed-region focus measurements for the V2 experiment.

The defaults are experimental parameters, not calibrated optical thresholds.
Sharpness is measured in normalized grayscale units; brightness stays in 0–255.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency on some hosts
    cv2 = None

from .models import FrameScore, ROI, TileScore


@dataclass(frozen=True)
class MetricConfig:
    """Initial candidates to calibrate with recorded sample images."""

    gaussian_sigma: float = 0.8
    dark_level: int = 5
    saturation_level: int = 250
    severe_clipped_fraction: float = 0.98
    min_valid_tiles: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.gaussian_sigma) or not 0 < self.gaussian_sigma <= 4:
            raise ValueError("gaussian_sigma must be finite and in (0, 4]")
        if not 0 <= self.dark_level < self.saturation_level <= 255:
            raise ValueError("invalid brightness thresholds")
        if not 0.5 < self.severe_clipped_fraction <= 1:
            raise ValueError("severe_clipped_fraction must be in (0.5, 1]")
        if type(self.min_valid_tiles) is not int or not 1 <= self.min_valid_tiles <= 4:
            raise ValueError("min_valid_tiles must be an integer from 1 to 4")


def evaluate_frame(image: np.ndarray, roi: ROI, config: MetricConfig | None = None) -> FrameScore:
    """Score four fixed subregions without resizing, tracking or interpolation.

    Low sharpness remains valid: it is evidence on a defocused side of a peak.
    Only near-total bright/dark clipping invalidates a tile. Unsupported image
    formats and out-of-bounds ROIs raise, instead of silently changing the ROI.
    """
    config = config or MetricConfig()
    if cv2 is None:
        raise RuntimeError("OpenCV is required for V2 image measurements")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("V2 measurements require a uint8 image")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("V2 measurements require grayscale or three-channel BGR images")
    roi.validate_shape(image.shape)
    selected = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    gray = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY) if selected.ndim == 3 else selected
    xs = (0, roi.width // 2, roi.width)
    ys = (0, roi.height // 2, roi.height)
    tiles = []
    for row in range(2):
        for col in range(2):
            x0, x1 = xs[col : col + 2]
            y0, y1 = ys[row : row + 2]
            patch = gray[y0:y1, x0:x1]
            normalized = patch.astype(np.float32) / 255.0
            smoothed = cv2.GaussianBlur(normalized, (0, 0), config.gaussian_sigma)
            gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
            gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
            # Laplacian intentionally retains the original pixels as a separate
            # comparison metric; it is never mixed into the primary score.
            laplacian = cv2.Laplacian(normalized, cv2.CV_32F)
            bright = float(np.mean(patch >= config.saturation_level))
            dark = float(np.mean(patch <= config.dark_level))
            reasons = []
            if bright >= config.severe_clipped_fraction:
                reasons.append("severe_overexposure")
            if dark >= config.severe_clipped_fraction:
                reasons.append("severe_underexposure")
            tiles.append(TileScore(
                bounds=(roi.x + x0, roi.y + y0, x1 - x0, y1 - y0),
                gradient=float(np.mean(gx * gx + gy * gy)),
                laplacian=float(np.var(laplacian)),
                mean_brightness=float(np.mean(patch)),
                saturation_fraction=bright,
                dark_fraction=dark,
                valid=not reasons,
                reasons=tuple(reasons),
                brightness_std=float(np.std(patch)),
            ))
    valid_scores = [tile.gradient for tile in tiles if tile.valid]
    valid = len(valid_scores) >= config.min_valid_tiles
    reasons = ("insufficient_exposure_valid_tiles",) if not valid else ()
    # Include the effective configuration so differently processed recordings
    # cannot silently be combined into one focus curve.
    method = (
        f"gradient_v1:sigma={config.gaussian_sigma:g}:dark={config.dark_level}"
        f":bright={config.saturation_level}:clipped={config.severe_clipped_fraction:g}"
        f":tiles={config.min_valid_tiles}"
    )
    return FrameScore(
        roi=roi,
        score=float(np.median(valid_scores)) if valid_scores else 0.0,
        tiles=tuple(tiles),
        valid=valid,
        reasons=reasons,
        method=method,
    )
