import json
import unittest

import numpy as np

from sample_plane import SamplePlanePoint, fit_sample_plane
from scan_plan import ScanBounds, generate_stitching_grid
from stitching_store import StitchingSessionStore, TileRecord


class ImageStitchingCoreTest(unittest.TestCase):
    def test_plane_fit_predicts_z_inside_four_focused_corners(self):
        corners = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c2", 100, 0, 110),
            SamplePlanePoint("c3", 100, 100, 130),
            SamplePlanePoint("c4", 0, 100, 120),
        ]

        plane = fit_sample_plane(corners)

        self.assertEqual(round(plane.z_at(50, 50)), 115)
        self.assertLessEqual(plane.max_abs_residual, 1e-6)

    def test_plane_fit_rejects_too_few_points(self):
        corners = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c2", 100, 0, 110),
        ]

        with self.assertRaisesRegex(ValueError, "at least three"):
            fit_sample_plane(corners)

    def test_generate_stitching_grid_uses_snake_order_and_plane_z(self):
        plane = fit_sample_plane(
            [
                SamplePlanePoint("c1", 0, 0, 100),
                SamplePlanePoint("c2", 100, 0, 100),
                SamplePlanePoint("c3", 100, 100, 120),
                SamplePlanePoint("c4", 0, 100, 120),
            ]
        )
        bounds = ScanBounds(min_x=0, max_x=100, min_y=0, max_y=100)

        tiles = generate_stitching_grid(bounds, rows=3, cols=3, plane=plane)

        self.assertEqual(
            [(tile.row, tile.col, tile.x, tile.y) for tile in tiles],
            [
                (0, 0, 0, 0),
                (0, 1, 50, 0),
                (0, 2, 100, 0),
                (1, 2, 100, 50),
                (1, 1, 50, 50),
                (1, 0, 0, 50),
                (2, 0, 0, 100),
                (2, 1, 50, 100),
                (2, 2, 100, 100),
            ],
        )
        self.assertEqual(tiles[4].z, 110)

    def test_stitching_store_writes_tile_image_and_metadata(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            store = StitchingSessionStore.create(Path(temp_dir), "demo")
            frame = np.full((4, 5, 3), 127, dtype=np.uint8)
            record = TileRecord(row=0, col=0, x=10, y=20, z=30, filename="", focus_score=12.5)

            saved = store.save_tile(frame, record)
            store.write_metadata(
                corners=[SamplePlanePoint("c1", 0, 0, 0)],
                tiles=[saved],
                settings={"rows": 1, "cols": 1},
            )

            self.assertTrue((store.path / saved.filename).exists())
            metadata = json.loads((store.path / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["settings"], {"rows": 1, "cols": 1})
            self.assertEqual(metadata["tiles"][0]["filename"], saved.filename)
            self.assertEqual(metadata["corners"][0]["label"], "c1")

    def test_stitching_store_uses_unique_folder_when_name_exists(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            first = StitchingSessionStore.create(root, "demo")
            second = StitchingSessionStore.create(root, "demo")

            self.assertEqual(first.path.name, "demo")
            self.assertEqual(second.path.name, "demo_01")


if __name__ == "__main__":
    unittest.main()
