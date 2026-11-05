"""Bounded stationary-burst Z search and independent return verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Callable

from .analysis import AnalysisConfig, analyze_bursts, summarize_burst
from .metrics import evaluate_frame
from .models import FrameRecord, Measurement, ROI
from .motion import ZMotion


@dataclass(frozen=True)
class AutofocusConfig:
    half_range: int = 20
    coarse_step: int = 5
    fine_step: int = 1
    speed: int = 10
    settle_seconds: float = 0.5
    frames: int = 5

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("half_range", 2, 200), ("coarse_step", 1, 20),
            ("fine_step", 1, 20), ("speed", 1, 20), ("frames", 3, 15),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if self.fine_step > self.coarse_step:
            raise ValueError("fine_step must not exceed coarse_step")
        if not math.isfinite(self.settle_seconds) or not 0 <= self.settle_seconds <= 5:
            raise ValueError("settle_seconds must be finite and in [0, 5]")
        if len(_grid(-self.half_range, self.half_range, self.coarse_step)) < 5:
            raise ValueError("coarse scan needs at least five sampled positions")
        estimated_points = 2 * self.half_range // self.coarse_step + 3 + 2 * self.coarse_step // self.fine_step + 3
        if estimated_points > 201:
            raise ValueError("scan settings exceed the 201-position experiment limit")


@dataclass(frozen=True)
class AutofocusResult:
    status: str
    message: str
    best_z: int | None = None
    measurements: tuple[Measurement, ...] = ()


def _grid(lower: int, upper: int, step: int) -> tuple[int, ...]:
    return tuple(sorted(set(range(lower, upper + 1, step)) | {lower, upper}))


def run_autofocus(
    controller,
    stream,
    roi: ROI,
    config: AutofocusConfig,
    cancel: threading.Event,
    progress: Callable[[str, object], None] | None = None,
    *,
    expected_start: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> AutofocusResult:
    """Search only the starting Z ± half_range and verify a measured return.

    Z comes from controller pulse counters. Host read freshness and arrival
    feedback support stationary sampling with uncalibrated camera latency.
    """
    measurements: list[Measurement] = []
    motion = None
    analysis_config = AnalysisConfig()

    def check_cancel() -> None:
        if cancel.is_set():
            raise InterruptedError("自动对焦已停止")

    def report(kind: str, payload: object) -> None:
        check_cancel()
        if progress is not None:
            progress(kind, payload)
        check_cancel()

    def result(status: str, message: str, best_z: int | None = None) -> AutofocusResult:
        return AutofocusResult(status, message, best_z, tuple(measurements))

    def sample(target: int, phase: str) -> Measurement:
        nonlocal expected_shape
        check_cancel()
        motion.move_to(target, cancel)
        check_cancel()
        if cancel.wait(config.settle_seconds):
            check_cancel()
        if motion.position() != target:
            raise RuntimeError("采样前控制器 Z 计数与目标不一致")
        frames = stream.collect(config.frames, discard=3, timeout=10.0, cancel=cancel)
        check_cancel()
        if motion.position() != target:
            raise RuntimeError("图像组采集期间控制器 Z 计数发生变化")
        records = []
        for frame in frames:
            shape = frame.image.shape
            if expected_shape is None:
                expected_shape = shape
            if shape != expected_shape:
                raise RuntimeError("采样期间相机图像尺寸发生变化")
            records.append(FrameRecord(
                frame.sequence, frame.read_started, frame.received_at, "",
                evaluate_frame(frame.image, roi),
            ))
        measurement = Measurement(
            f"{phase}-{len(measurements):03d}", float(target), "controller_counter", tuple(records),
            notes=f"{phase}; stationary burst; host read timestamps",
        )
        measurements.append(measurement)
        report("measurement", measurement)
        return measurement

    try:
        check_cancel()
        latest = stream.latest()
        if not stream.active or latest is None:
            return result("not_ready", "请先打开相机并等待曝光调节完成")
        if expected_shape is not None and latest.image.shape != expected_shape:
            return result("not_ready", "相机图像尺寸已偏离确认时的画面，请重新选择区域并确认")
        expected_shape = latest.image.shape
        roi.validate_shape(expected_shape)
        # Resolve the initial counter through the same idle checks used during search.
        motion = ZMotion(controller, -0x7FFFFFFF, 0x7FFFFFFF, config.speed)
        origin = motion.position()
        if expected_start is not None and origin != expected_start:
            return result("not_ready", "当前 Z 已偏离确认的起始位置，请重新确认扫描范围")
        lower, upper = origin - config.half_range, origin + config.half_range
        motion = ZMotion(controller, lower, upper, config.speed)
        report("status", f"预检 Z={origin}；扫描范围 [{lower}, {upper}] pulse")
        initial = sample(origin, "preflight")
        if not summarize_burst(initial).valid:
            motion.stop()
            return result("invalid_measurements", "起始图像组曝光或有效帧不足，请检查照明和样品区域")

        report("status", "粗扫描中：逐点停稳后采集图像组")
        coarse = [sample(target, "coarse") for target in _grid(lower, upper, config.coarse_step)]
        coarse_analysis = analyze_bursts(coarse, analysis_config)
        if coarse_analysis.status != "candidate_only":
            motion.stop()
            return result(coarse_analysis.status, coarse_analysis.message)

        bracket_lower, bracket_upper = (int(value) for value in coarse_analysis.peak_interval)
        fine_targets = _grid(bracket_lower, bracket_upper, config.fine_step)
        if len(measurements) + len(fine_targets) + 1 > 201:
            motion.stop()
            return result("scan_limit", "候选峰区过宽，细扫描将超过 201 位置实验上限")
        report("status", f"细扫描中：[{bracket_lower}, {bracket_upper}] pulse")
        fine = [sample(target, "fine") for target in fine_targets]
        # Both passes remain evidence. Inconsistent revisits therefore prevent success.
        combined = coarse + fine
        analysis = analyze_bursts(combined, analysis_config)
        if analysis.status != "candidate_only":
            motion.stop()
            return result(analysis.status, analysis.message)

        best = int(analysis.best_z)
        # Both scan passes approach sampled points in increasing Z. Revisit from
        # below inside the original bounds to keep the final approach consistent.
        approach = max(lower, best - config.fine_step)
        if approach >= best:
            motion.stop()
            return result("verification_failed", "候选点缺少范围内的同向回位空间")
        report("status", f"回到候选 Z={best}，执行独立图像组验证")
        motion.move_to(approach, cancel)
        verification = sample(best, "verification")
        verified_analysis = analyze_bursts(combined + [verification], analysis_config)
        if verified_analysis.status != "candidate_only" or int(verified_analysis.best_z) != best:
            motion.stop()
            return result("verification_failed", f"回位验证未通过：{verified_analysis.message}")
        reference = next(point for point in analysis.points if point.z == best)
        observed = summarize_burst(verification, analysis_config)
        common_tiles = tuple(index for index in range(4) if all(point.tile_valid[index] for point in analysis.points))
        if not observed.valid or any(
            not observed.tile_valid[index]
            or abs(observed.tile_scores[index] - reference.tile_scores[index])
            > 0.30 * max(reference.tile_scores[index], 1e-12)
            for index in common_tiles
        ):
            motion.stop()
            return result("verification_failed", "回位后的清晰度与扫描候选偏差超过实验容限")
        check_cancel()
        return result("verified", f"已回到 Z={best} pulse，独立图像组验证通过", best)
    except InterruptedError:
        status, message = "cancelled", "自动对焦已停止，当前停留位置需现场确认"
    except Exception as exc:
        status, message = "error", f"自动对焦中止：{exc}"
    try:
        if motion is not None:
            motion.stop()
        else:
            controller.emergency_stop_all()
    except Exception as exc:
        message += f"；软件急停发送失败：{exc}"
    return result(status, message)
