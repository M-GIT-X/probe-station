from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gui_app import ProbeStationApp, StitchingFocusMode
from sample_plane import SamplePlanePoint
from scan_plan import TilePoint


class FakeIncrementalMosaicBuilder:
    writes = []

    def __init__(self, *_args, **_kwargs):
        type(self).writes = []
        self.added = []

    def add_tile(self, index, frame):
        self.added.append((index, frame.shape))

    def write(self, output):
        type(self).writes.append(Path(output).name)
        return Path(output)

    def write_with_boundaries(self, output):
        type(self).writes.append(Path(output).name)
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
    def test_stitching_capture_scores_stabilized_frame_but_returns_raw_frame(self):
        raw_frame = np.full((6, 6, 3), 40, dtype=np.uint8)
        stabilized_frame = np.full((6, 6, 3), 200, dtype=np.uint8)
        fake_app = SimpleNamespace(
            camera=SimpleNamespace(read_frame=Mock(return_value=raw_frame)),
            camera_lock=threading.Lock(),
            _raise_if_stitching_stopped=lambda: None,
        )

        with (
            patch("gui_app.stabilize_frame_translation", return_value=(stabilized_frame, {"applied": True})),
            patch("gui_app.auto_select_rois", return_value=[(0, 0, 6, 6)]),
            patch("gui_app.calculate_focus_index", return_value=321.0),
            patch(
                "gui_app.brightness_diagnostics",
                return_value={
                    "mean_brightness": 200.0,
                    "frame_saturation": 0.0,
                    "underexposed_fraction": 0.0,
                    "overexposed": False,
                },
            ),
        ):
            frame, score = ProbeStationApp._capture_stable_stitching_frame(fake_app, 1)

        np.testing.assert_array_equal(frame, raw_frame)
        self.assertEqual(score, 321.0)
        self.assertTrue(fake_app._last_stitching_capture_info["saved_from_raw_frame"])
        self.assertEqual(fake_app._last_stitching_capture_info["mean_brightness"], 200.0)

    def test_scan_worker_full_auto_focuses_three_by_three_probe_grid_for_plane(self):
        events = []
        focus_calls = []

        def focus_corner(corner, speed, settle_seconds, sample_frames):
            del speed, settle_seconds, sample_frames
            focus_calls.append((corner.label, corner.x, corner.y))
            z = corner.x + corner.y
            if corner.label == "p22":
                z += 80
            return SamplePlanePoint(corner.label, corner.x, corner.y, z)

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
        plan = SimpleNamespace(rows=1, cols=1, tiles=[TilePoint(row=0, col=0, x=5, y=5, z=10)])
        store = Mock(path=Path("/tmp/session"))
        store.save_tile.side_effect = lambda _frame, record: record
        captured_plane_z = {}

        def make_plan(bounds, plane, **kwargs):
            del bounds, kwargs
            captured_plane_z["center"] = plane.z_at(5, 5)
            return plan

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
            patch("gui_app.generate_overlap_scan_plan", side_effect=make_plan),
            patch("gui_app.IncrementalMosaicBuilder", FakeIncrementalMosaicBuilder),
            patch("gui_app.stitch_session_by_metadata", side_effect=AssertionError("offline stitch should not run")),
        ):
            ProbeStationApp._stitching_scan_worker(
                fake_app,
                corners=corners,
                plane=None,
                bounds=SimpleNamespace(min_x=0, max_x=10, min_y=0, max_y=10),
                focus_mode=StitchingFocusMode.FULL.value,
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        self.assertEqual(
            focus_calls,
            [
                ("p11", 0, 0),
                ("p12", 5, 0),
                ("p13", 10, 0),
                ("p21", 0, 5),
                ("p22", 5, 5),
                ("p23", 10, 5),
                ("p31", 0, 10),
                ("p32", 5, 10),
                ("p33", 10, 10),
            ],
        )
        self.assertEqual(captured_plane_z["center"], 10)
        written_corners = store.write_metadata.call_args.kwargs["corners"]
        self.assertEqual([corner.z for corner in written_corners], [0, 0, 0, 0])
        plane_validation = store.write_metadata.call_args.kwargs["settings"]["plane_validation"]
        self.assertEqual([point["label"] for point in plane_validation["outliers"]], ["p22"])
        store.write_tile_quality_csv.assert_called_once()
        self.assertEqual(
            FakeIncrementalMosaicBuilder.writes,
            ["mechanical_only_mosaic.png", "mosaic_with_boundaries.png", "stitched_mosaic.png"],
        )

    def test_scan_worker_semi_auto_uses_manually_focused_corner_z_values(self):
        events = []
        focus_corner = Mock(side_effect=AssertionError("Semi Auto should not refocus recorded corners"))
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
            SamplePlanePoint("c1", 0, 0, 11),
            SamplePlanePoint("c2", 10, 0, 22),
            SamplePlanePoint("c3", 10, 10, 33),
            SamplePlanePoint("c4", 0, 10, 44),
        ]
        point = SimpleNamespace(x=0, y=0, z=0)
        trial = SimpleNamespace(
            x_step_pulses=5,
            y_step_pulses=5,
            reference=point,
            x_trial=point,
            y_trial=point,
        )
        plan = SimpleNamespace(rows=1, cols=1, tiles=[TilePoint(row=0, col=0, x=0, y=0, z=11)])
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
                focus_mode=StitchingFocusMode.SEMI.value,
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        focus_corner.assert_not_called()
        written_corners = store.write_metadata.call_args.kwargs["corners"]
        self.assertEqual([corner.z for corner in written_corners], [11, 22, 33, 44])

    def test_scan_worker_semi_auto_records_plane_confirmation_without_refocusing_from_geometry(self):
        events = []
        focus_calls = []

        def focus_corner(corner, speed, settle_seconds, sample_frames):
            del speed, settle_seconds, sample_frames
            focus_calls.append(corner.label)
            return corner

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
            SamplePlanePoint("c3", 10, 10, 80),
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
                focus_mode=StitchingFocusMode.SEMI.value,
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        self.assertEqual(focus_calls, [])
        written_corners = store.write_metadata.call_args.kwargs["corners"]
        self.assertEqual([corner.z for corner in written_corners], [0, 0, 80, 0])
        plane_validation = store.write_metadata.call_args.kwargs["settings"]["plane_validation"]
        self.assertFalse(plane_validation["accepted"])
