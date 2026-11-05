from collections import deque
from concurrent.futures import Future
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autofocus_v2.models import ROI
import trial_gui


class EmbeddedJogTest(unittest.TestCase):
    def setUp(self):
        self.root = Mock()
        self.target = Mock()
        self.controller = Mock()
        self.workers = Mock()
        with patch("z_jog.ThreadPoolExecutor", return_value=self.workers):
            self.jog = trial_gui.EmbeddedJog(self.root, Mock(), self.controller, self.target)

    def test_input_field_never_dispatches_motion_after_arming(self):
        self.jog._set_armed(True)
        self.root.focus_get.return_value = object()
        self.jog._press_in_panel(SimpleNamespace(keysym="w"))
        self.workers.submit.assert_not_called()

    def test_focus_out_of_entry_does_not_disarm_target(self):
        self.jog._set_armed(True)
        self.jog._focus_out(SimpleNamespace(widget=object()))
        self.assertTrue(self.jog._armed)
        self.jog._focus_out(SimpleNamespace(widget=self.target))
        self.assertFalse(self.jog._armed)

    def test_emergency_stop_is_available_with_entry_focused(self):
        self.root.focus_get.return_value = object()
        self.assertEqual(self.jog._press_in_panel(SimpleNamespace(keysym="space")), "break")
        self.assertTrue(self.jog._cancel.is_set())
        self.workers.submit.assert_called_once_with(self.controller.emergency_stop_all)

    def test_cross_axis_press_during_motion_is_dropped_and_stop_cancels(self):
        self.root.focus_get.return_value = self.target
        move, stop = Future(), Future()
        self.workers.submit.side_effect = [move, stop]
        self.jog._set_armed(True)
        self.jog._press_in_panel(SimpleNamespace(keysym="w"))
        self.jog._press_in_panel(SimpleNamespace(keysym="d"))
        self.assertEqual(self.workers.submit.call_count, 1)
        self.jog._request_stop()
        self.assertTrue(self.jog._cancel.is_set())
        self.workers.submit.assert_called_with(self.controller.emergency_stop_all)
        move.set_result(None)
        stop.set_result(None)
        self.jog._poll()
        self.jog._press_in_panel(SimpleNamespace(keysym="q"))
        self.assertEqual(self.workers.submit.call_count, 2)

    def test_autofocus_gate_blocks_arming_and_movement(self):
        self.root.focus_get.return_value = self.target
        self.jog.blocked = lambda: True
        self.jog._set_armed(True)
        self.jog._press_in_panel(SimpleNamespace(keysym='w'))
        self.assertFalse(self.jog._armed)
        self.workers.submit.assert_not_called()

    def test_stop_latches_autofocus_before_dispatching_emergency(self):
        autofocus_cancel = threading.Event()
        self.jog.on_stop = autofocus_cancel.set
        self.workers.submit.side_effect = lambda *_args: self.assertTrue(autofocus_cancel.is_set())
        self.jog._request_stop()
        self.assertTrue(self.jog._cancel.is_set())


class TrialCameraTest(unittest.TestCase):
    def test_startup_does_not_open_camera_or_controller(self):
        root = Mock()
        with patch.object(trial_gui.TrialApp, "_build"), patch.object(trial_gui, "FrameStream") as stream, patch.object(
            trial_gui, "StageController"
        ) as stage:
            app = trial_gui.TrialApp(root)
        stream.return_value.start.assert_not_called()
        stage.assert_not_called()
        self.assertIsNone(app.controller)

    def test_new_selection_maps_original_pixels_and_clears_curve(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.shape = (400, 800)
        app.af_running = False
        app.display_size = (400, 200)
        app.drag_start = (20, 30)
        app.history = deque([(1, 0.4)])
        app.preview = Mock()
        app.camera_status = Mock()
        app._draw_plot = Mock()
        app._draw_roi = Mock()
        app._drag_end(SimpleNamespace(x=120, y=100))
        self.assertEqual(app.roi, ROI(40, 60, 200, 140))
        self.assertEqual(list(app.history), [])

    def test_invalid_sample_breaks_plot_instead_of_joining_lines(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.plot = Mock()
        app.history = deque([(0, 1), (1, 2), (2, None), (3, 3), (4, 4)])
        app._draw_plot()
        # One axes call and two disjoint curve segments.
        self.assertEqual(app.plot.create_line.call_count, 3)

    def test_invalid_frame_adds_a_gap_and_reuses_selected_roi(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.shape = (120, 160)
        app.roi = ROI(10, 20, 32, 32)
        app.history = deque([(1, 0.1)])
        app.tile_mask = (True, True, True, True)
        app.preview = Mock()
        app.preview.winfo_width.return_value = 160
        app.preview.winfo_height.return_value = 120
        app.score_status = Mock()
        app.root = Mock()
        app._draw_plot = Mock()
        app._draw_roi = Mock()
        frame = SimpleNamespace(image=np.zeros((120, 160, 3), dtype=np.uint8), received_at=2)
        result = SimpleNamespace(valid=False, score=0.0, tiles=[SimpleNamespace(valid=True)] * 4)
        with patch.object(trial_gui, "_score_frame", return_value=result) as score, patch.object(
            trial_gui.tk, "PhotoImage"
        ) as photo:
            app._show_frame(frame)
        score.assert_called_once_with(frame.image, ROI(10, 20, 32, 32))
        self.assertEqual(list(app.history), [(1, 0.1), (2, None)])
        self.assertTrue(photo.call_args.kwargs["data"].startswith(b"P6\n160 120\n255\n"))

    def test_tile_validity_change_clears_previous_score_history(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.shape = (120, 160)
        app.roi = ROI(10, 20, 32, 32)
        app.history = deque([(1, 0.1)])
        app.tile_mask = (True, True, True, True)
        app.preview = Mock()
        app.preview.winfo_width.return_value = 160
        app.preview.winfo_height.return_value = 120
        app.score_status = Mock()
        app.root = Mock()
        app._draw_plot = Mock()
        app._draw_roi = Mock()
        frame = SimpleNamespace(image=np.zeros((120, 160, 3), dtype=np.uint8), received_at=2)
        tiles = [SimpleNamespace(valid=value) for value in (True, True, True, False)]
        with patch.object(trial_gui, "_score_frame", return_value=SimpleNamespace(valid=True, score=0.2, tiles=tiles)), patch.object(
            trial_gui.tk, "PhotoImage"
        ):
            app._show_frame(frame)
        self.assertEqual(list(app.history), [(2, 0.2)])
        app.score_status.set.assert_called_with("V2 0.2 · 3/4 区块有效")


class AutofocusLifecycleTest(unittest.TestCase):
    def test_confirmed_start_passes_frozen_position_and_shape_to_engine(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.root = Mock()
        app.controller = Mock()
        app.stream = Mock(active=True)
        app.stream.latest.return_value = SimpleNamespace(image=np.zeros((64, 64, 3), dtype=np.uint8))
        app.af_queue = queue.Queue()
        roi = ROI(0, 0, 32, 32)
        config = trial_gui.AutofocusConfig()
        app.af_queue.put(('prepared', (100, roi, config, (64, 64, 3))))
        app.closing = threading.Event()
        app.af_cancel = threading.Event()
        app.af_status = Mock()
        app._finish_autofocus = Mock()

        def thread_factory(*, target, daemon):
            thread = Mock()
            thread.start.side_effect = target
            return thread

        result = SimpleNamespace(status='verified', message='已验证')
        with patch.object(trial_gui.messagebox, 'askyesno', return_value=True), patch.object(
            trial_gui.threading, 'Thread', side_effect=thread_factory,
        ), patch.object(trial_gui, 'run_autofocus', autospec=True, return_value=result) as engine:
            app._poll_autofocus()
        engine.assert_called_once()
        self.assertEqual(engine.call_args.args, (app.controller, app.stream, roi, config, app.af_cancel))
        self.assertEqual(engine.call_args.kwargs['expected_start'], 100)
        self.assertEqual(engine.call_args.kwargs['expected_shape'], (64, 64, 3))
        app._finish_autofocus.assert_called_once_with('verified · 已验证')

    def test_close_waits_for_autofocus_and_pending_emergency_stop(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.root = Mock()
        app.closing = threading.Event()
        app.closing.set()
        app.connection_thread = None
        app.af_thread = Mock()
        app.af_thread.is_alive.return_value = True
        app.jog = SimpleNamespace(_move=None, _stop=Future())
        app._poll_autofocus = Mock()
        app._tick()
        app.root.quit.assert_not_called()
        app.af_thread.is_alive.return_value = False
        app._tick()
        app.root.quit.assert_not_called()
        app.jog._stop = None
        app._tick()
        app.root.quit.assert_called_once()

    def test_cancel_during_preparation_prevents_confirmation_and_launch(self):
        app = trial_gui.TrialApp.__new__(trial_gui.TrialApp)
        app.af_queue = queue.Queue()
        app.af_queue.put(('prepared', (100, ROI(0, 0, 32, 32), Mock(), (64, 64, 3))))
        app.closing = threading.Event()
        app.af_cancel = threading.Event()
        app.af_cancel.set()
        app._finish_autofocus = Mock()
        with patch.object(trial_gui.messagebox, 'askyesno') as confirm, patch.object(
            trial_gui.threading, 'Thread'
        ) as thread:
            app._poll_autofocus()
        confirm.assert_not_called()
        thread.assert_not_called()
        app._finish_autofocus.assert_called_once()


if __name__ == "__main__":
    unittest.main()
