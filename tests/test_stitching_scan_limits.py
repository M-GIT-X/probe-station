from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gui_app import ProbeStationApp
from sample_plane import SamplePlanePoint
from scan_plan import TilePoint


class FakeIncrementalMosaicBuilder:
    def __init__(self, *_args, **_kwargs):
        self.added = []

    def add_tile(self, index, frame):
        self.added.append((index, frame.shape))

    def write(self, output):
        return Path(output)

    def close(self):
        return None


class StitchingScanLimitTest(unittest.TestCase):
    def test_scan_worker_accepts_automatic_plan_larger_than_100_tiles(self):
        events = []
        fake_app = SimpleNamespace(
            device_queue=SimpleNamespace(put=events.append),
            controller=SimpleNamespace(is_open=True, stop_all=Mock()),
            stage_image_calibration=None,
            _raise_if_stitching_stopped=lambda: None,
            _move_to_absolute_position=lambda *_args: None,
            _sleep_with_stitching_stop=lambda *_args: None,
            _capture_stable_stitching_frame=lambda _count: (np.zeros((4, 4, 3), dtype=np.uint8), 1.0),
            _focus_stitching_corner=lambda corner, *_args: corner,
        )
        corners = [
            SamplePlanePoint("c1", 0, 0, 0),
            SamplePlanePoint("c2", 10, 0, 0),
            SamplePlanePoint("c3", 10, 10, 0),
            SamplePlanePoint("c4", 0, 10, 0),
        ]
        point = SimpleNamespace(x=0, y=0, z=0)
        trial = SimpleNamespace(
            x_step_pulses=5,
            y_step_pulses=5,
            reference=point,
            x_trial=point,
            y_trial=point,
        )
        plan = SimpleNamespace(
            rows=1,
            cols=101,
            tiles=[TilePoint(row=0, col=col, x=col, y=0, z=0) for col in range(101)],
        )
        store = Mock(path=Path("/tmp/session"))
        store.save_tile.side_effect = lambda _frame, record: record

        with (
            patch("gui_app.StitchingSessionStore.create", return_value=store),
            patch("gui_app.boundary_polygon_from_points", return_value=[]),
            patch("gui_app.build_in_bounds_trial_plan", return_value=trial),
            patch(
                "gui_app.estimate_calibration_from_frames",
                return_value=SimpleNamespace(
                    frame_width=100,
                    frame_height=100,
                    x_pixels_per_pulse=1.0,
                    y_pixels_per_pulse=1.0,
                ),
            ),
            patch("gui_app.generate_overlap_scan_plan", return_value=plan),
            patch("gui_app.IncrementalMosaicBuilder", FakeIncrementalMosaicBuilder),
            patch("gui_app.stitch_session_by_metadata", side_effect=AssertionError("offline stitch should not run")) as offline_stitch,
        ):
            ProbeStationApp._stitching_scan_worker(
                fake_app,
                corners=corners,
                plane=None,
                bounds=SimpleNamespace(),
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        self.assertEqual(store.save_tile.call_count, 101)
        offline_stitch.assert_not_called()
        self.assertFalse(any(kind == "stitch_error" for kind, _payload in events))


if __name__ == "__main__":
    unittest.main()


class StitchingCornerAutofocusTest(unittest.TestCase):
    def test_scan_worker_refocuses_recorded_xy_corners_before_fitting_plane(self):
        events = []
        focused_z_values = iter([10, 20, 30, 40])

        def focus_corner(corner, speed, settle_seconds, sample_frames):
            del speed, settle_seconds, sample_frames
            return SamplePlanePoint(corner.label, corner.x, corner.y, next(focused_z_values))

        fake_app = SimpleNamespace(
            device_queue=SimpleNamespace(put=events.append),
            controller=SimpleNamespace(is_open=True, stop_all=Mock()),
            stage_image_calibration=None,
            _raise_if_stitching_stopped=lambda: None,
            _move_to_absolute_position=lambda *_args: None,
            _sleep_with_stitching_stop=lambda *_args: None,
            _capture_stable_stitching_frame=lambda _count: (np.zeros((4, 4, 3), dtype=np.uint8), 1.0),
            _focus_stitching_corner=focus_corner,
        )
        corners = [
            SamplePlanePoint("c1", 0, 0, 0),
            SamplePlanePoint("c2", 10, 0, 0),
            SamplePlanePoint("c3", 10, 10, 0),
            SamplePlanePoint("c4", 0, 10, 0),
        ]
        point = SimpleNamespace(x=0, y=0, z=0)
        trial = SimpleNamespace(
            x_step_pulses=5,
            y_step_pulses=5,
            reference=point,
            x_trial=point,
            y_trial=point,
        )
        plan = SimpleNamespace(rows=1, cols=1, tiles=[TilePoint(row=0, col=0, x=0, y=0, z=10)])
        store = Mock(path=Path("/tmp/session"))
        store.save_tile.side_effect = lambda _frame, record: record

        with (
            patch("gui_app.StitchingSessionStore.create", return_value=store),
            patch("gui_app.build_in_bounds_trial_plan", return_value=trial),
            patch(
                "gui_app.estimate_calibration_from_frames",
                return_value=SimpleNamespace(
                    frame_width=100,
                    frame_height=100,
                    x_pixels_per_pulse=1.0,
                    y_pixels_per_pulse=1.0,
                ),
            ),
            patch("gui_app.generate_overlap_scan_plan", return_value=plan),
            patch("gui_app.IncrementalMosaicBuilder", FakeIncrementalMosaicBuilder),
            patch("gui_app.stitch_session_by_metadata", side_effect=AssertionError("offline stitch should not run")),
        ):
            ProbeStationApp._stitching_scan_worker(
                fake_app,
                corners=corners,
                plane=None,
                bounds=SimpleNamespace(min_x=0, max_x=10, min_y=0, max_y=10),
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        written_corners = store.write_metadata.call_args.kwargs["corners"]
        self.assertEqual([corner.z for corner in written_corners], [10, 20, 30, 40])

    def test_scan_worker_retries_all_corners_once_when_plane_confirmation_is_ambiguous(self):
        events = []
        focused_z_values = iter([0, 0, 80, 0, 0, 0, 0, 0])
        focus_calls = []

        def focus_corner(corner, speed, settle_seconds, sample_frames):
            del speed, settle_seconds, sample_frames
            focus_calls.append(corner.label)
            return SamplePlanePoint(corner.label, corner.x, corner.y, next(focused_z_values))

        fake_app = SimpleNamespace(
            device_queue=SimpleNamespace(put=events.append),
            controller=SimpleNamespace(is_open=True, stop_all=Mock()),
            stage_image_calibration=None,
            _raise_if_stitching_stopped=lambda: None,
            _move_to_absolute_position=lambda *_args: None,
            _sleep_with_stitching_stop=lambda *_args: None,
            _capture_stable_stitching_frame=lambda _count: (np.zeros((4, 4, 3), dtype=np.uint8), 1.0),
            _focus_stitching_corner=focus_corner,
        )
        corners = [
            SamplePlanePoint("c1", 0, 0, 0),
            SamplePlanePoint("c2", 10, 0, 0),
            SamplePlanePoint("c3", 10, 10, 0),
            SamplePlanePoint("c4", 0, 10, 0),
        ]
        point = SimpleNamespace(x=0, y=0, z=0)
        trial = SimpleNamespace(
            x_step_pulses=5,
            y_step_pulses=5,
            reference=point,
            x_trial=point,
            y_trial=point,
        )
        plan = SimpleNamespace(rows=1, cols=1, tiles=[TilePoint(row=0, col=0, x=0, y=0, z=0)])
        store = Mock(path=Path("/tmp/session"))
        store.save_tile.side_effect = lambda _frame, record: record

        with (
            patch("gui_app.StitchingSessionStore.create", return_value=store),
            patch("gui_app.build_in_bounds_trial_plan", return_value=trial),
            patch(
                "gui_app.estimate_calibration_from_frames",
                return_value=SimpleNamespace(
                    frame_width=100,
                    frame_height=100,
                    x_pixels_per_pulse=1.0,
                    y_pixels_per_pulse=1.0,
                ),
            ),
            patch("gui_app.generate_overlap_scan_plan", return_value=plan),
            patch("gui_app.IncrementalMosaicBuilder", FakeIncrementalMosaicBuilder),
            patch("gui_app.stitch_session_by_metadata", side_effect=AssertionError("offline stitch should not run")),
        ):
            ProbeStationApp._stitching_scan_worker(
                fake_app,
                corners=corners,
                plane=None,
                bounds=SimpleNamespace(min_x=0, max_x=10, min_y=0, max_y=10),
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        self.assertEqual(focus_calls, ["c1", "c2", "c3", "c4", "c1", "c2", "c3", "c4"])
        written_corners = store.write_metadata.call_args.kwargs["corners"]
        self.assertEqual([corner.z for corner in written_corners], [0, 0, 0, 0])

    def test_scan_worker_stops_when_corner_plane_confirmation_still_fails_after_retry(self):
        events = []

        def focus_corner(corner, speed, settle_seconds, sample_frames):
            del speed, settle_seconds, sample_frames
            z_values = {"c1": 0, "c2": 0, "c3": 80, "c4": 0}
            return SamplePlanePoint(corner.label, corner.x, corner.y, z_values[corner.label])

        fake_app = SimpleNamespace(
            device_queue=SimpleNamespace(put=events.append),
            controller=SimpleNamespace(is_open=True, stop_all=Mock()),
            stage_image_calibration=None,
            _raise_if_stitching_stopped=lambda: None,
            _move_to_absolute_position=lambda *_args: None,
            _sleep_with_stitching_stop=lambda *_args: None,
            _capture_stable_stitching_frame=lambda _count: (np.zeros((4, 4, 3), dtype=np.uint8), 1.0),
            _focus_stitching_corner=focus_corner,
        )
        corners = [
            SamplePlanePoint("c1", 0, 0, 0),
            SamplePlanePoint("c2", 10, 0, 0),
            SamplePlanePoint("c3", 10, 10, 0),
            SamplePlanePoint("c4", 0, 10, 0),
        ]

        with patch("gui_app.StitchingSessionStore.create", return_value=Mock(path=Path("/tmp/session"))):
            ProbeStationApp._stitching_scan_worker(
                fake_app,
                corners=corners,
                plane=None,
                bounds=SimpleNamespace(min_x=0, max_x=10, min_y=0, max_y=10),
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        errors = [payload for kind, payload in events if kind == "stitch_error"]
        self.assertTrue(errors)
        self.assertIn("focused stitching corners do not define a reliable plane", errors[0])
