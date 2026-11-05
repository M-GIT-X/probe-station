from dataclasses import replace
import unittest

import numpy as np

from autofocus_v2.analysis import AnalysisConfig, analyze_bursts, summarize_burst
from autofocus_v2 import metrics
from autofocus_v2.metrics import evaluate_frame
from autofocus_v2.models import FrameRecord, FrameScore, Measurement, ROI, TileScore


ROI_VALUE = ROI(0, 0, 32, 32)


def measurement(z, scores, *, tiles=None, brightness=128.0, sequence_offset=0):
    frames = []
    for index, score in enumerate(scores):
        tile_values = tiles[index] if tiles is not None else (score,) * 4
        tile_scores = tuple(TileScore(
            bounds=((tile_index % 2) * 16, (tile_index // 2) * 16, 16, 16),
            gradient=float(value), laplacian=float(value), mean_brightness=brightness,
            saturation_fraction=0.0, dark_fraction=0.0, valid=True,
        ) for tile_index, value in enumerate(tile_values))
        frames.append(FrameRecord(
            sequence=index + sequence_offset,
            read_started=1000 + z * 10 + index + sequence_offset,
            received_at=1000.1 + z * 10 + index + sequence_offset,
            image_path=f"{z}_{sequence_offset}_{index}.png",
            metrics=FrameScore(ROI_VALUE, float(score), tile_scores, True),
        ))
    return Measurement(f"point-{z}-{sequence_offset}", float(z), "operator_entered", tuple(frames))


def curve(values, **kwargs):
    return [measurement(z, [value] * 5, **kwargs) for z, value in enumerate(values)]


class BurstSummaryTest(unittest.TestCase):
    def test_retains_median_spread_and_all_frame_count(self):
        summary = summarize_burst(measurement(0, [2, 4, 6, 8, 100]))
        self.assertEqual(summary.score, 6)
        self.assertEqual(summary.spread, 4)
        self.assertEqual(summary.frame_count, 5)
        self.assertEqual(summary.valid_frame_count, 5)
        self.assertEqual(summary.tile_spreads, (4, 4, 4, 4))
        self.assertTrue(summary.valid)

    def test_two_frames_do_not_prove_a_stationary_burst(self):
        summary = summarize_burst(measurement(0, [8, 8]))
        self.assertFalse(summary.valid)
        self.assertIn("insufficient_valid_frames", summary.reasons)

    def test_duplicate_frame_is_not_an_independent_sample(self):
        original = measurement(0, [8, 8, 8])
        summary = summarize_burst(replace(original, frames=original.frames + original.frames[:1]))
        self.assertFalse(summary.valid)
        self.assertIn("duplicate_frames", summary.reasons)

    def test_nan_scores_cannot_enter_summary(self):
        original = measurement(0, [1, 1, 1])
        first = original.frames[0]
        bad = replace(first, metrics=replace(first.metrics, score=float("nan")))
        summary = summarize_burst(replace(original, frames=(bad,) + original.frames[1:]))
        self.assertFalse(summary.valid)
        self.assertIn("invalid_metric_values", summary.reasons)
        self.assertTrue(np.isfinite(summary.score))

    def test_zero_is_valid_data(self):
        summary = summarize_burst(measurement(0, [0, 0, 0]))
        self.assertTrue(summary.valid)
        self.assertEqual(summary.score, 0)


class FocusAnalysisTest(unittest.TestCase):
    def test_clear_peak_is_only_a_candidate_at_a_measured_z(self):
        result = analyze_bursts(curve([1, 4, 10, 4, 1]))
        self.assertEqual(result.status, "candidate_only")
        self.assertEqual(result.best_z, 2)
        self.assertEqual(result.peak_interval, (1, 3))
        self.assertTrue(result.warnings)

    def test_empty_and_too_few_positions(self):
        for values in ([], [1], [1, 3, 1], [1, 3, 6, 1]):
            with self.subTest(values=values):
                self.assertEqual(analyze_bursts(curve(values)).status, "insufficient_data")

    def test_zero_and_flat_never_produce_candidate(self):
        self.assertEqual(analyze_bursts(curve([0] * 5)).status, "insufficient_texture")
        self.assertEqual(analyze_bursts(curve([10, 10.2, 10.5, 10.2, 10])).status, "flat_curve")

    def test_monotonic_and_boundary_plateau_require_other_side(self):
        for values in ([1, 2, 4, 6, 10], [10, 10, 4, 2, 1], [1, 4, 8, 10, 10]):
            with self.subTest(values=values):
                self.assertEqual(analyze_bursts(curve(values)).status, "boundary_peak")

    def test_separated_competing_peaks_are_reported(self):
        result = analyze_bursts(curve([1, 9, 3, 10, 1]))
        self.assertEqual(result.status, "multiple_peaks")
        self.assertIsNone(result.best_z)

    def test_small_shoulder_is_not_a_competing_peak(self):
        result = analyze_bursts(curve([1, 9, 8.7, 10, 1]))
        self.assertEqual(result.status, "candidate_only")
        self.assertEqual(result.best_z, 3)

    def test_prominent_lower_peak_cannot_validate_an_unbracketed_global_maximum(self):
        # The scan ends while still near its highest score. A separate, lower
        # peak does not supply the missing evidence on the highest peak's right.
        result = analyze_bursts(curve([1, 9, 1, 10, 9.9]))
        self.assertEqual(result.status, "weak_peak")
        self.assertIsNone(result.best_z)
        self.assertIsNone(result.peak_interval)

    def test_internal_plateau_keeps_measured_bracket(self):
        result = analyze_bursts(curve([1, 5, 10, 10, 5, 1]))
        self.assertEqual(result.status, "candidate_only")
        self.assertIn(result.best_z, (2, 3))
        self.assertEqual(result.peak_interval, (1, 4))

    def test_large_frame_variation_is_not_hidden_by_median(self):
        measurements = curve([1, 4, 10, 4, 1])
        measurements[2] = measurement(2, [2, 5, 10, 15, 18])
        self.assertEqual(analyze_bursts(measurements).status, "unstable_measurements")

    def test_tile_fluctuation_is_not_hidden_by_stable_frame_median(self):
        measurements = curve([1, 4, 10, 4, 1])
        measurements[2] = measurement(2, [10] * 5, tiles=[(v, 10, 10, 10) for v in (2, 5, 10, 15, 18)])
        self.assertEqual(analyze_bursts(measurements).status, "unstable_measurements")

    def test_different_tile_focal_planes_are_not_fused(self):
        a = [1, 10, 6, 2, 1]
        b = [1, 2, 6, 10, 1]
        c = [1, 4, 10, 4, 1]
        measurements = [measurement(z, [float(np.median([a[z], b[z], c[z], c[z]]))] * 5, tiles=[(a[z], b[z], c[z], c[z])] * 5) for z in range(5)]
        self.assertEqual(analyze_bursts(measurements).status, "roi_conflict")

    def test_lower_tile_peak_does_not_hide_unbracketed_maximum_at_another_depth(self):
        a = [1, 9, 1, 10, 9.9]
        b = [1, 10, 5, 3, 1]
        measurements = [measurement(z, [b[z]] * 5, tiles=[(a[z], b[z], b[z], b[z])] * 5) for z in range(5)]
        self.assertEqual(analyze_bursts(measurements).status, "roi_conflict")

    def test_one_informative_tile_is_insufficient(self):
        measurements = [measurement(z, [value] * 5, tiles=[(value, 0, 0, 0)] * 5) for z, value in enumerate([1, 4, 10, 4, 1])]
        self.assertEqual(analyze_bursts(measurements).status, "insufficient_texture")

    def test_inconsistent_roi_or_method_is_not_combined(self):
        for changed in ({"roi": ROI(1, 0, 32, 32)}, {"method": "different"}):
            measurements = curve([1, 4, 10, 4, 1])
            first = measurements[0].frames[0]
            measurements[0] = replace(measurements[0], frames=(replace(first, metrics=replace(first.metrics, **changed)),) + measurements[0].frames[1:])
            with self.subTest(changed=changed):
                self.assertEqual(analyze_bursts(measurements).status, "incompatible_measurements")

    def test_nonfinite_z_and_mixed_sources_are_rejected(self):
        measurements = curve([1, 4, 10, 4, 1])
        bad_z = [replace(measurements[0], z=float("nan"))] + measurements[1:]
        self.assertEqual(analyze_bursts(bad_z).status, "invalid_measurements")
        mixed = [replace(measurements[0], z_source="controller")] + measurements[1:]
        self.assertEqual(analyze_bursts(mixed).status, "incompatible_measurements")

    def test_duplicates_at_same_z_are_pooled_not_best_selected(self):
        measurements = curve([1, 4, 10, 4, 1])
        measurements.append(measurement(2, [2] * 5, sequence_offset=50))
        result = analyze_bursts(measurements)
        self.assertEqual(len(result.points), 5)
        self.assertEqual(result.points[2].frame_count, 10)
        self.assertEqual(result.points[2].score, 6)
        self.assertEqual(result.status, "unstable_measurements")

    def test_short_contradictory_revisit_is_not_hidden_by_a_long_burst(self):
        measurements = curve([1, 4, 10, 4, 1])
        measurements[2] = measurement(2, [10] * 20)
        measurements.append(measurement(2, [2] * 3, sequence_offset=50))
        result = analyze_bursts(measurements)
        self.assertEqual(result.points[2].score, 10)
        self.assertEqual(result.points[2].spread, 0)
        self.assertEqual(result.status, "unstable_measurements")

    def test_short_brightness_changed_revisit_is_not_hidden_by_pooling(self):
        measurements = curve([1, 4, 10, 4, 1])
        measurements[2] = measurement(2, [10] * 20)
        measurements.append(measurement(2, [10] * 3, brightness=170, sequence_offset=50))
        self.assertEqual(analyze_bursts(measurements).status, "exposure_changed")

    def test_exposure_change_between_positions_is_detected(self):
        measurements = curve([1, 4, 10, 4, 1])
        measurements[2] = measurement(2, [10] * 5, brightness=170)
        self.assertEqual(analyze_bursts(measurements).status, "exposure_changed")

    def test_local_brightness_change_is_not_hidden_by_other_tiles(self):
        for levels, expected in (([170] * 5, "exposure_changed"), ([95, 105, 128, 150, 160], "unstable_measurements")):
            measurements = curve([1, 4, 10, 4, 1])
            changed = []
            for frame, brightness in zip(measurements[2].frames, levels):
                tiles = (replace(frame.metrics.tiles[0], mean_brightness=brightness),) + frame.metrics.tiles[1:]
                changed.append(replace(frame, metrics=replace(frame.metrics, tiles=tiles)))
            measurements[2] = replace(measurements[2], frames=tuple(changed))
            with self.subTest(levels=levels):
                self.assertEqual(analyze_bursts(measurements).status, expected)

    def test_invalid_subtile_is_excluded_at_all_positions(self):
        measurements = curve([1, 4, 10, 4, 1])
        frames = []
        for frame in measurements[0].frames:
            tiles = (replace(frame.metrics.tiles[0], valid=False, reasons=("severe_overexposure",)),) + frame.metrics.tiles[1:]
            frames.append(replace(frame, metrics=replace(frame.metrics, tiles=tiles)))
        measurements[0] = replace(measurements[0], frames=tuple(frames))
        result = analyze_bursts(measurements)
        self.assertEqual(result.status, "candidate_only")
        self.assertTrue(any("3 个" in warning for warning in result.warnings))

    def test_negative_or_nonfinite_config_is_rejected(self):
        for kwargs in ({"min_frames": 1}, {"min_positions": 2}, {"min_valid_tiles": 1}, {"max_relative_spread": float("nan")}, {"max_brightness_spread": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AnalysisConfig(**kwargs)


@unittest.skipIf(metrics.cv2 is None, "OpenCV is not installed")
class ImageFocusAnalysisTest(unittest.TestCase):
    def image_measurement(self, z, images):
        roi = ROI(0, 0, 128, 128)
        frames = tuple(FrameRecord(
            sequence=z * 10 + index,
            read_started=1000 + z * 10 + index,
            received_at=1000.1 + z * 10 + index,
            image_path=f"synthetic-{z}-{index}.png",
            metrics=evaluate_frame(image, roi),
        ) for index, image in enumerate(images))
        return Measurement(f"synthetic-{z}", float(z), "manual_annotation", frames)

    def test_noisy_blurred_images_recover_known_sampled_focus_as_candidate_only(self):
        for seed in (6, 813, 2026):
            rng = np.random.default_rng(seed)
            texture = rng.integers(40, 215, (128, 128)).astype(np.float32)
            measurements = []
            for z in range(7):
                blurred = metrics.cv2.GaussianBlur(texture, (0, 0), 0.5 + 0.7 * abs(z - 3))
                images = [np.clip(blurred + rng.normal(0, 0.5, texture.shape), 0, 255).astype(np.uint8) for _ in range(5)]
                measurements.append(self.image_measurement(z, images))
            result = analyze_bursts(measurements)
            with self.subTest(seed=seed):
                self.assertEqual(result.status, "candidate_only")
                self.assertEqual(result.best_z, 3)
                self.assertEqual(result.peak_interval, (2, 4))

    def test_disappearing_reflection_cannot_manufacture_a_focus_peak(self):
        rng = np.random.default_rng(29)
        texture = rng.integers(120, 137, (128, 128), dtype=np.uint8)
        fine_detail = rng.integers(20, 235, (64, 64), dtype=np.uint8)
        measurements = []
        for z in range(5):
            image = texture.copy()
            image[:64, :64] = fine_detail if z == 2 else 255
            measurements.append(self.image_measurement(z, [image] * 5))
        result = analyze_bursts(measurements)
        self.assertEqual(result.status, "flat_curve")
        self.assertIsNone(result.best_z)
        self.assertTrue(any("3 个" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
