"""Conservative analysis of manually labelled stationary image bursts.

This module finds measured candidate brackets, never an autofocus success or a
motor target. Thresholds below are experimental defaults pending real samples.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .models import BurstSummary, FocusAnalysis, FrameRecord, Measurement


@dataclass(frozen=True)
class AnalysisConfig:
    min_frames: int = 3
    min_positions: int = 5
    min_valid_tiles: int = 2
    min_valid_fraction: float = 0.75
    min_texture_score: float = 1e-8
    min_relative_prominence: float = 0.15
    secondary_peak_ratio: float = 0.8
    max_relative_spread: float = 0.30
    max_brightness_spread: float = 15.0
    max_brightness_drift: float = 20.0
    roi_peak_tolerance_steps: float = 1.0

    def __post_init__(self) -> None:
        for name, minimum in (("min_frames", 3), ("min_positions", 3), ("min_valid_tiles", 2)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.min_valid_tiles > 4:
            raise ValueError("min_valid_tiles cannot exceed four fixed tiles")
        for name in ("min_valid_fraction", "min_relative_prominence", "secondary_peak_ratio", "max_relative_spread"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("min_texture_score", "max_brightness_spread", "max_brightness_drift", "roi_peak_tolerance_steps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _iqr(values: Sequence[float]) -> float:
    return float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else 0.0


def _frame_well_formed(frame: FrameRecord) -> bool:
    metrics = frame.metrics
    if len(metrics.tiles) != 4 or not math.isfinite(metrics.score) or metrics.score < 0:
        return False
    return all(
        all(math.isfinite(value) for value in (
            tile.gradient, tile.laplacian, tile.mean_brightness,
            tile.saturation_fraction, tile.dark_fraction, tile.brightness_std,
        )) and tile.gradient >= 0 and tile.laplacian >= 0
        for tile in metrics.tiles
    )


def _summarize(measurement: Measurement, config: AnalysisConfig, indices: tuple[int, ...] | None = None) -> BurstSummary:
    frames = measurement.frames
    usable = [frame for frame in frames if frame.metrics.valid and _frame_well_formed(frame)]
    reasons = []
    signatures = {(frame.sequence, frame.read_started, frame.received_at) for frame in frames}
    if len(signatures) != len(frames):
        reasons.append("duplicate_frames")
    if any(
        not math.isfinite(frame.read_started) or not math.isfinite(frame.received_at)
        or frame.read_started > frame.received_at
        for frame in frames
    ):
        reasons.append("invalid_frame_times")
    if any(not _frame_well_formed(frame) for frame in frames):
        reasons.append("invalid_metric_values")
    if frames and any(
        frame.metrics.roi != frames[0].metrics.roi
        or frame.metrics.method != frames[0].metrics.method
        or tuple(t.bounds for t in frame.metrics.tiles) != tuple(t.bounds for t in frames[0].metrics.tiles)
        for frame in frames
    ):
        reasons.append("inconsistent_roi_or_method")
    tile_values = [[frame.metrics.tiles[index].gradient for frame in usable if frame.metrics.tiles[index].valid] for index in range(4)]
    tile_valid = tuple(
        len(values) >= config.min_frames and len(values) / max(1, len(frames)) >= config.min_valid_fraction
        for values in tile_values
    )
    selected = indices if indices is not None else tuple(i for i, valid in enumerate(tile_valid) if valid)
    joint_frames = [frame for frame in usable if selected and all(frame.metrics.tiles[i].valid for i in selected)]
    scores = [float(np.median([frame.metrics.tiles[i].gradient for i in selected])) for frame in joint_frames]
    brightness = [float(np.median([frame.metrics.tiles[i].mean_brightness for i in selected])) for frame in joint_frames]
    if len(selected) < config.min_valid_tiles:
        reasons.append("insufficient_valid_tiles")
    if len(joint_frames) < config.min_frames:
        reasons.append("insufficient_valid_frames")
    if len(joint_frames) / max(1, len(frames)) < config.min_valid_fraction:
        reasons.append("too_many_invalid_frames")
    return BurstSummary(
        z=float(measurement.z),
        score=float(np.median(scores)) if scores else 0.0,
        spread=_iqr(scores),
        tile_scores=tuple(float(np.median(values)) if values else 0.0 for values in tile_values),
        frame_count=len(frames),
        valid=not reasons,
        reasons=tuple(reasons),
        tile_spreads=tuple(_iqr(values) for values in tile_values),
        brightness_spread=_iqr(brightness),
        valid_frame_count=len(joint_frames),
        tile_valid=tile_valid,
    )


def summarize_burst(measurement: Measurement, config: AnalysisConfig | None = None) -> BurstSummary:
    """Retain medians and IQRs; these are spread diagnostics, not confidence intervals."""
    return _summarize(measurement, config or AnalysisConfig())


def _peaks(values: np.ndarray) -> list[tuple[int, int, float]]:
    """Interior peak plateaus with topographic prominence, without smoothing."""
    peaks = []
    index = 1
    while index < len(values) - 1:
        end = index
        while end + 1 < len(values) and values[end + 1] == values[index]:
            end += 1
        if end < len(values) - 1 and values[index] > values[index - 1] and values[end] > values[end + 1]:
            height = values[index]
            left = index - 1
            while left > 0 and values[left] <= height:
                left -= 1
            right = end + 1
            while right < len(values) - 1 and values[right] <= height:
                right += 1
            floor = max(float(np.min(values[left : index + 1])), float(np.min(values[end : right + 1])))
            peaks.append((index, end, float(height - floor)))
        index = end + 1
    return peaks


def analyze_bursts(measurements: Sequence[Measurement], config: AnalysisConfig | None = None) -> FocusAnalysis:
    """Find a bracket only when fixed tiles support a repeatable interior peak.

    Repeated Z labels pool all frames, while keeping each burst's stability
    evidence. Z labels remain operator evidence; no interpolation turns receive
    timestamps into exposure positions.
    """
    config = config or AnalysisConfig()
    warning_list = [
        "候选来自已采样图像；尚未执行运动、回位或独立对焦验证。",
        "Z 标签和静止状态由操作者提供；收帧时间不是曝光时间。",
        "判据为实验默认值，仍需真实样品标定。",
    ]
    points: tuple[BurstSummary, ...] = ()

    def result(status: str, message: str, best_z: float | None = None, interval: tuple[float, float] | None = None) -> FocusAnalysis:
        return FocusAnalysis(status, message, points, best_z, interval, tuple(warning_list))

    if not measurements:
        return result("insufficient_data", "尚无测量点；请在多个已知 Z 位置采集静止图像组。")
    if any(not math.isfinite(m.z) or not m.z_source for m in measurements):
        return result("invalid_measurements", "Z 位置必须为有限数值，且注明位置来源。")
    sources = {m.z_source for m in measurements}
    if len(sources) != 1:
        return result("incompatible_measurements", "位置来源不同，不能直接合并比较。")
    all_frames = [frame for measurement in measurements for frame in measurement.frames]
    if all_frames:
        first = all_frames[0].metrics
        if any(
            frame.metrics.roi != first.roi or frame.metrics.method != first.method
            or tuple(t.bounds for t in frame.metrics.tiles) != tuple(t.bounds for t in first.tiles)
            for frame in all_frames
        ):
            return result("incompatible_measurements", "目标区域或评分版本发生变化，请分别分析。")
    grouped: dict[float, list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.z].append(measurement)
    merged = [Measurement(
        sample_id="+".join(m.sample_id for m in group),
        z=z,
        z_source=group[0].z_source,
        frames=tuple(frame for m in group for frame in m.frames),
    ) for z, group in sorted(grouped.items())]
    if len(merged) < len(measurements):
        warning_list.append("相同 Z 标签的重复测量已合并全部帧，并保留其波动。")
    points = tuple(summarize_burst(m, config) for m in merged)
    if any(not point.valid for point in points):
        return result("invalid_measurements", "部分图像组缺少足够有效帧或具有不一致记录，需补采这些位置。")
    common_tiles = tuple(index for index in range(4) if all(point.tile_valid[index] for point in points))
    if len(common_tiles) < config.min_valid_tiles:
        return result("insufficient_valid_tiles", "各 Z 位置之间没有足够的共同有效子区域。")
    points = tuple(_summarize(m, config, common_tiles) for m in merged)
    if any(not point.valid for point in points):
        return result("invalid_measurements", "共同子区域缺少同时有效的图像帧。")
    if len(common_tiles) < 4:
        warning_list.append(f"仅使用各位置共同有效的 {len(common_tiles)} 个固定子区域比较。")
    if len(points) < config.min_positions:
        return result("insufficient_data", f"至少需要 {config.min_positions} 个不同 Z 位置，目前有 {len(points)} 个。")
    values = np.asarray([point.score for point in points], dtype=float)
    z_values = np.asarray([point.z for point in points], dtype=float)
    maximum = float(np.max(values))
    if maximum <= config.min_texture_score:
        return result("insufficient_texture", "所有位置清晰度信息都很弱；无法确定焦点。")
    original_bursts = tuple(_summarize(m, config, common_tiles) for m in measurements)
    if any(
        point.spread > config.max_relative_spread * max(point.score, maximum * 0.1)
        or point.brightness_spread > config.max_brightness_spread
        or any(point.tile_spreads[i] > config.max_relative_spread * max(point.tile_scores[i], maximum * 0.1) for i in common_tiles)
        for point in (*points, *original_bursts)
    ):
        return result("unstable_measurements", "帧间清晰度或亮度波动过大；请停稳并检查成像条件后重采。")
    # Pooling a long burst with a short, contradictory revisit can make the
    # shorter burst disappear from both the pooled median and its IQR. Compare
    # burst medians as well, so a repeat at the same Z remains evidence.
    for point in points:
        repeats = [burst for burst in original_bursts if burst.z == point.z and burst.valid_frame_count]
        if len(repeats) > 1 and any(
            np.ptp([burst.tile_scores[i] for burst in repeats])
            > config.max_relative_spread * max(point.tile_scores[i], maximum * 0.1)
            for i in common_tiles
        ):
            return result("unstable_measurements", "相同 Z 的重复采集清晰度不一致；需检查回位、漂移或成像变化。")
    brightness = []
    for measurement in measurements:
        tile_brightness = np.asarray([
            [frame.metrics.tiles[i].mean_brightness for i in common_tiles]
            for frame in measurement.frames
            if frame.metrics.valid and _frame_well_formed(frame)
            and all(frame.metrics.tiles[i].valid for i in common_tiles)
        ])
        if not len(tile_brightness):
            continue
        # A median over tiles can hide illumination changes in one part of the
        # selected sample; each fixed tile must retain comparable brightness.
        if np.any(np.subtract(*np.percentile(tile_brightness, [75, 25], axis=0)) > config.max_brightness_spread):
            return result("unstable_measurements", "子区域帧间亮度波动过大；请检查成像条件后重采。")
        brightness.append(np.median(tile_brightness, axis=0))
    if np.any(np.ptp(brightness, axis=0) > config.max_brightness_drift):
        return result("exposure_changed", "采样期间固定子区域亮度变化明显；当前分数不能可靠归因于焦距。")
    if (maximum - float(np.min(values))) / maximum < config.min_relative_prominence:
        return result("flat_curve", "曲线变化太小，尚无可辨别的焦峰。")
    best_indices = np.flatnonzero(values == maximum)
    if 0 in best_indices or len(values) - 1 in best_indices:
        return result("boundary_peak", "最高分位于采样边界，需要在允许范围内补充峰另一侧的信息。")
    significant = [
        (start, end) for start, end, prominence in _peaks(values)
        if values[start] >= config.secondary_peak_ratio * maximum
        and prominence >= config.min_relative_prominence * maximum
    ]
    if len(significant) > 1:
        return result("multiple_peaks", "曲线存在多个有竞争力的焦峰，需要补测或重新选择目标区域。")
    if not significant:
        return result("weak_peak", "最高分两侧下降证据不足，暂不能圈定可靠峰区。")
    start, end = significant[0]
    if not all(start <= index <= end for index in best_indices):
        return result("weak_peak", "最高分两侧下降证据不足，暂不能圈定可靠峰区。")
    informative_peaks = []
    for tile_index in common_tiles:
        tile_values = np.asarray([point.tile_scores[tile_index] for point in points])
        tile_max = float(np.max(tile_values))
        if tile_max <= config.min_texture_score or np.ptp(tile_values) / tile_max < config.min_relative_prominence:
            continue
        tile_peaks = [
            (start, end) for start, end, prominence in _peaks(tile_values)
            if tile_values[start] >= config.secondary_peak_ratio * tile_max
            and prominence >= config.min_relative_prominence * tile_max
        ]
        if tile_values[0] == tile_max or tile_values[-1] == tile_max or len(tile_peaks) != 1:
            return result("roi_conflict", "不同子区域对焦峰位置或峰形给出了不一致证据。")
        start, end = tile_peaks[0]
        if not all(start <= index <= end for index in np.flatnonzero(tile_values == tile_max)):
            return result("roi_conflict", "子区域的最高分缺少两侧下降证据，不能用其较低峰支持候选焦位。")
        informative_peaks.append((float(z_values[start]), float(z_values[end])))
    if len(informative_peaks) < config.min_valid_tiles:
        return result("insufficient_texture", "具有明确焦峰的子区域不足，需补采或选择更有纹理的区域。")
    tolerance = config.roi_peak_tolerance_steps * float(np.median(np.diff(z_values)))
    if max(start for start, _ in informative_peaks) - min(end for _, end in informative_peaks) > tolerance:
        return result("roi_conflict", "子区域的最佳焦位分离过大，可能存在不同高度、漂移或测量冲突。")
    start, end = significant[0]
    best_index = int(best_indices[len(best_indices) // 2])
    # The interval names the neighbouring measured Z positions, not a fitted
    # uncertainty bound; sampling gaps may still hide narrower peaks.
    interval = (float(z_values[start - 1]), float(z_values[end + 1]))
    warning_list.append("峰区为相邻采样位置构成的区间，不能排除采样间隙中的更窄焦峰。")
    return result("candidate_only", "已找到待验证的候选峰区；这不代表自动对焦完成。", float(z_values[best_index]), interval)
