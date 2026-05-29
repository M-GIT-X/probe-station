import unittest

from gui_app import preview_tile_draw_indices
from sample_plane import SamplePlanePoint
from scan_plan import TilePoint
from stitching_view_model import build_stitching_view_model


class StitchingViewModelTest(unittest.TestCase):
    def test_view_model_normalizes_corners_tiles_and_current_position(self):
        corners = [
            SamplePlanePoint("c1", 10, 20, 100),
            SamplePlanePoint("c2", 110, 20, 100),
            SamplePlanePoint("c3", 110, 120, 120),
            SamplePlanePoint("c4", 10, 120, 120),
        ]
        tiles = [
            TilePoint(row=0, col=0, x=10, y=20, z=100),
            TilePoint(row=0, col=1, x=110, y=20, z=100),
        ]

        model = build_stitching_view_model(corners, tiles, current_tile_index=1)

        self.assertEqual(model.corners[0].label, "c1")
        self.assertEqual((model.corners[0].nx, model.corners[0].ny), (0.0, 0.0))
        self.assertEqual((model.corners[2].nx, model.corners[2].ny), (1.0, 1.0))
        self.assertEqual(model.tiles[1].state, "current")
        self.assertEqual(model.tiles[0].state, "done")

    def test_completed_capture_count_keeps_all_saved_tiles_marked_done(self):
        corners = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c2", 100, 0, 100),
            SamplePlanePoint("c3", 100, 100, 100),
            SamplePlanePoint("c4", 0, 100, 100),
        ]
        tiles = [
            TilePoint(row=0, col=0, x=0, y=0, z=100),
            TilePoint(row=0, col=1, x=100, y=0, z=100),
        ]

        model = build_stitching_view_model(corners, tiles, captured_tile_count=2)

        self.assertEqual([tile.state for tile in model.tiles], ["done", "done"])

    def test_preview_tile_draw_indices_draws_all_small_plans(self):
        self.assertEqual(preview_tile_draw_indices(4, sample_target=800), {0, 1, 2, 3})

    def test_preview_tile_draw_indices_samples_large_plans_and_keeps_progress(self):
        indices = preview_tile_draw_indices(
            5000,
            current_tile_index=1234,
            captured_tile_count=3456,
            sample_target=800,
        )

        self.assertIn(0, indices)
        self.assertIn(4999, indices)
        self.assertIn(1234, indices)
        self.assertIn(3455, indices)
        self.assertLessEqual(len(indices), 804)


if __name__ == "__main__":
    unittest.main()
