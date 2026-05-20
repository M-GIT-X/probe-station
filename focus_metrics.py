"""Focus-quality metrics for manual focus assist."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local optional dependency
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is required for focus metrics")
    return cv2


def auto_select_rois(frame) -> list[tuple[int, int, int, int]]:
    cv = _require_cv2()
    height, width = frame.shape[:2]
    roi_w = max(32, width // 7)
    roi_h = max(32, height // 7)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for row in range(5):
        for col in range(7):
            cx = int((col + 0.5) * width / 7)
            cy = int((row + 0.5) * height / 5)
            x = min(max(0, cx - roi_w // 2), max(0, width - roi_w))
            y = min(max(0, cy - roi_h // 2), max(0, height - roi_h))
            patch = gray[y : y + roi_h, x : x + roi_w]
            if patch.size == 0:
                continue

            contrast = float(patch.std())
            brightness = float(patch.mean())
            lap_var = float(cv.Laplacian(patch, cv.CV_64F).var())
            edges = cv.Canny(patch, 60, 140)
            edge_density = float(np.mean(edges > 0))
            saturation = 0.0
            if frame.ndim == 3:
                color_patch = frame[y : y + roi_h, x : x + roi_w]
                saturation = float(np.mean((color_patch <= 3) | (color_patch >= 252)))

            brightness_score = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0)
            saturation_score = max(0.0, 1.0 - saturation * 2.0)
            quality = (
                contrast * 1.4
                + lap_var * 0.08
                + edge_density * 180.0
                + brightness_score * 18.0
                + saturation_score * 12.0
            )
            candidates.append((quality, (x, y, roi_w, roi_h)))

    selected: list[tuple[int, int, int, int]] = []
    for _quality, roi in sorted(candidates, reverse=True):
        if len(selected) >= 8:
            break
        if all(_overlap_ratio(roi, existing) < 0.35 for existing in selected):
            selected.append(roi)

    if len(selected) < 6:
        for _quality, roi in sorted(candidates, reverse=True):
            if roi not in selected:
                selected.append(roi)
            if len(selected) >= 6:
                break
    return selected[:8]


def calculate_focus_index(frame, rois: list[tuple[int, int, int, int]] | None = None) -> float:
    if frame is None:
        return 0.0
    cv = _require_cv2()
    if rois is None:
        rois = auto_select_rois(frame)

    values: list[float] = []
    for x, y, width, height in rois:
        patch = frame[y : y + height, x : x + width]
        if patch.size == 0:
            continue
        gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY) if patch.ndim == 3 else patch

        laplacian_var = float(cv.Laplacian(gray, cv.CV_64F).var())
        sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))
        contrast = float(gray.std())
        brightness = float(gray.mean())

        brightness_penalty = 1.0
        if brightness < 35.0:
            brightness_penalty = max(0.2, brightness / 35.0)
        elif brightness > 225.0:
            brightness_penalty = max(0.2, (255.0 - brightness) / 30.0)

        saturation_penalty = 1.0
        if patch.ndim == 3:
            saturated_ratio = float(np.mean((patch <= 3) | (patch >= 252)))
            saturation_penalty = max(0.2, 1.0 - saturated_ratio * 2.0)

        contrast_factor = min(1.5, max(0.35, contrast / 28.0))
        score = (0.72 * laplacian_var + 0.018 * tenengrad + 1.8 * contrast)
        values.append(score * contrast_factor * brightness_penalty * saturation_penalty)

    return robust_representative(values)


def brightness_diagnostics(frame) -> dict[str, float | bool]:
    if frame is None:
        return {
            "frame_saturation": 0.0,
            "overexposed": False,
            "mean_brightness": 0.0,
            "underexposed_fraction": 0.0,
        }
    arr = np.asarray(frame)
    frame_saturation = float(np.mean(arr >= 250))
    mean_brightness = float(arr.mean())
    underexposed_fraction = float(np.mean(arr <= 5))
    return {
        "frame_saturation": frame_saturation,
        "overexposed": frame_saturation > 0.10 or mean_brightness > 220.0,
        "mean_brightness": mean_brightness,
        "underexposed_fraction": underexposed_fraction,
    }


def calculate_focus_index_with_info(
    frame,
    rois: list[tuple[int, int, int, int]] | None = None,
) -> tuple[float, dict[str, float | bool]]:
    score = calculate_focus_index(frame, rois)
    info = brightness_diagnostics(frame)
    info["focus_index"] = float(score)
    return score, info


def robust_representative(values) -> float:
    cleaned = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if cleaned.size == 0:
        return 0.0
    if cleaned.size < 5:
        return float(np.median(cleaned))
    cleaned.sort()
    lo = int(cleaned.size * 0.2)
    hi = max(lo + 1, int(cleaned.size * 0.8))
    return float(np.mean(cleaned[lo:hi]))


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min(aw * ah, bw * bh)
    return intersection / max(1, smaller)
