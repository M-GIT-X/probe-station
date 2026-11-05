import threading
import unittest
from unittest.mock import Mock

from autofocus_v2.motion import ZMotion
from stage_controller import StageControllerError


class ZMotionTest(unittest.TestCase):
    def setUp(self):
        self.controller = Mock(invert_z=False)
        self.motion = ZMotion(self.controller, -20, 20, 10)
        self.cancel = threading.Event()

    def test_move_requires_strict_arrival_and_target_counter(self):
        self.controller.read_axis_state.side_effect = [
            {"position": 0, "running": False}, {"position": 5, "running": False},
        ]
        self.motion.move_to(5, self.cancel)
        self.controller.move_relative.assert_called_once_with(
            "Z", 1, 5, 10, cancel=self.cancel, require_arrival_feedback=True,
        )

    def test_outside_target_and_cancellation_send_no_move(self):
        with self.assertRaises(ValueError):
            self.motion.move_to(21, self.cancel)
        self.cancel.set()
        with self.assertRaises(InterruptedError):
            self.motion.move_to(5, self.cancel)
        self.controller.move_relative.assert_not_called()

    def test_running_counter_mismatch_or_external_move_rejects(self):
        self.controller.read_axis_state.return_value = {"position": 0, "running": True}
        with self.assertRaisesRegex(StageControllerError, "still reports motion"):
            self.motion.position()
        self.controller.read_axis_state.side_effect = [
            {"position": 0, "running": False}, {"position": 4, "running": False},
        ]
        with self.assertRaisesRegex(StageControllerError, "does not match target"):
            self.motion.move_to(5, self.cancel)
        self.controller.read_axis_state.side_effect = None
        self.controller.read_axis_state.return_value = {"position": 0, "running": False}
        self.motion.position()
        self.controller.read_axis_state.return_value = {"position": 1, "running": False}
        with self.assertRaisesRegex(StageControllerError, "changed outside"):
            self.motion.position()

    def test_cancel_during_write_does_not_accept_arrival(self):
        self.controller.read_axis_state.return_value = {"position": 0, "running": False}
        self.controller.move_relative.side_effect = lambda *args, **kwargs: self.cancel.set()
        with self.assertRaises(InterruptedError):
            self.motion.move_to(5, self.cancel)
        self.assertEqual(self.controller.read_axis_state.call_count, 1)
