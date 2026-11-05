from collections import defaultdict
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autofocus_v2.autofocus import AutofocusConfig, run_autofocus
from autofocus_v2.models import CapturedFrame, ROI


class FakeController:
    def __init__(self):
        self.z = 0
        self.moves = []
        self.stops = 0

    def emergency_stop_all(self):
        self.stops += 1


class FakeMotion:
    def __init__(self, controller, lower, upper, speed):
        self.controller = controller
        self.lower, self.upper = lower, upper

    def position(self):
        return self.controller.z

    def move_to(self, target, cancel):
        if cancel.is_set():
            raise InterruptedError()
        assert self.lower <= target <= self.upper
        self.controller.z = target
        self.controller.moves.append(target)

    def stop(self):
        self.controller.emergency_stop_all()


class SyntheticStream:
    active = True

    def __init__(self, controller, optimum=2, flat=False, fail_return=False, dark=False):
        self.controller = controller
        self.optimum = optimum
        self.flat = flat
        self.fail_return = fail_return
        self.dark = dark
        self.reads = 0
        self.visits = defaultdict(int)
        rng = np.random.default_rng(730)
        self.texture = rng.integers(32, 224, size=(96, 96), dtype=np.uint8)

    def latest(self):
        return CapturedFrame(0, 0.0, 0.01, self.texture)

    def collect(self, count, *, discard, timeout, cancel):
        z = self.controller.z
        self.visits[z] += 1
        sigma = 0.0 if self.flat else abs(z - self.optimum) * 0.3
        if self.fail_return and z == self.optimum and self.visits[z] > 1:
            sigma = 5.0
        image = cv2.GaussianBlur(self.texture, (0, 0), sigma) if sigma else self.texture
        if self.dark:
            image = np.zeros_like(image)
        frames = []
        for _ in range(count):
            self.reads += 1
            frames.append(CapturedFrame(self.reads, 1000.0 + self.reads, 1000.01 + self.reads, image.copy()))
        return tuple(frames)


class AutofocusSearchTest(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.config = AutofocusConfig(half_range=10, coarse_step=5, settle_seconds=0)
        self.cancel = threading.Event()
        self.roi = ROI(8, 8, 80, 80)

    def run_search(self, stream, **kwargs):
        with patch("autofocus_v2.autofocus.ZMotion", FakeMotion):
            return run_autofocus(self.controller, stream, self.roi, self.config, self.cancel, **kwargs)

    def test_real_v2_scores_find_and_verify_synthetic_peak(self):
        result = self.run_search(SyntheticStream(self.controller))
        self.assertEqual(result.status, "verified", result.message)
        self.assertEqual(result.best_z, 2)
        self.assertEqual(self.controller.z, 2)
        self.assertEqual(self.controller.moves[-2:], [1, 2])
        self.assertTrue(all(-10 <= z <= 10 for z in self.controller.moves))
        self.assertEqual(result.measurements[-1].notes.split(";")[0], "verification")
        self.assertTrue(all(m.z_source == "controller_counter" for m in result.measurements))

    def test_boundary_and_flat_curves_do_not_return_a_success(self):
        for optimum, flat, expected in ((10, False, "boundary_peak"), (2, True, "flat_curve")):
            with self.subTest(expected=expected):
                self.controller = FakeController()
                result = self.run_search(SyntheticStream(self.controller, optimum, flat))
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.best_z)
                self.assertGreater(self.controller.stops, 0)

    def test_dark_preflight_stops_before_scanning(self):
        result = self.run_search(SyntheticStream(self.controller, dark=True))
        self.assertEqual(result.status, "invalid_measurements")
        self.assertEqual(self.controller.moves, [0])

    def test_independent_return_failure_is_reported(self):
        result = self.run_search(SyntheticStream(self.controller, fail_return=True))
        self.assertEqual(result.status, "verification_failed")
        self.assertIsNone(result.best_z)
        self.assertGreater(self.controller.stops, 0)

    def test_cancel_from_progress_prevents_any_next_move(self):
        def progress(kind, payload):
            if kind == "measurement":
                self.cancel.set()

        result = self.run_search(SyntheticStream(self.controller), progress=progress)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(self.controller.moves, [0])
        self.assertEqual(self.controller.stops, 1)

    def test_changed_confirmation_start_has_no_motion(self):
        result = self.run_search(SyntheticStream(self.controller), expected_start=1)
        self.assertEqual(result.status, "not_ready")
        self.assertEqual(self.controller.moves, [])

    def test_changed_confirmation_shape_has_no_motion(self):
        result = self.run_search(SyntheticStream(self.controller), expected_shape=(96, 96, 3))
        self.assertEqual(result.status, "not_ready")
        self.assertEqual(self.controller.moves, [])

    def test_capture_failure_stops_and_does_not_return_to_candidate(self):
        stream = SyntheticStream(self.controller)
        stream.collect = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lost camera"))
        result = self.run_search(stream)
        self.assertEqual(result.status, "error")
        self.assertIn("lost camera", result.message)
        self.assertEqual(self.controller.moves, [0])
        self.assertEqual(self.controller.stops, 1)


class AutofocusConfigTest(unittest.TestCase):
    def test_invalid_or_unbounded_inputs_are_rejected(self):
        for changed in (
            {"half_range": 201}, {"coarse_step": 0}, {"fine_step": 6},
            {"speed": 21}, {"frames": 2}, {"settle_seconds": float("nan")},
            {"half_range": 200, "coarse_step": 1}, {"half_range": 2, "coarse_step": 20},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                AutofocusConfig(**changed)
