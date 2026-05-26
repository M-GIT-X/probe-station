from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gui_app import ProbeStationApp
from scan_plan import TilePoint


class StitchingScanLimitTest(unittest.TestCase):
    def test_scan_worker_accepts_automatic_plan_larger_than_100_tiles(self):
        events = []
        fake_app = SimpleNamespace(
            device_queue=SimpleNamespace(put=events.append),
            controller=SimpleNamespace(is_open=True, stop_all=Mock()),
            _raise_if_stitching_stopped=lambda: None,
            _move_to_absolute_position=lambda *_args: None,
            _sleep_with_stitching_stop=lambda *_args: None,
            _capture_stable_stitching_frame=lambda _count: (np.zeros((4, 4, 3), dtype=np.uint8), 1.0),
        )
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
            patch("gui_app.stitch_session_by_metadata", return_value=Path("/tmp/session/mosaic.png")),
        ):
            ProbeStationApp._stitching_scan_worker(
                fake_app,
                corners=[],
                plane=SimpleNamespace(),
                bounds=SimpleNamespace(),
                overlap_percent=25.0,
                speed=2,
                settle_seconds=0.0,
                sample_frames=1,
                output_root=Path("/tmp"),
            )

        self.assertEqual(store.save_tile.call_count, 101)
        self.assertFalse(any(kind == "stitch_error" for kind, _payload in events))


if __name__ == "__main__":
    unittest.main()
