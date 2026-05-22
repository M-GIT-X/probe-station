import tempfile
from pathlib import Path
import unittest

import numpy as np

from image_stitcher import (
    refine_tile_positions_by_overlap,
    stitch_session_by_metadata,
    stitch_tiles_by_stage_coordinates,
)
from stitching_store import TileRecord


class ImageStitcherTest(unittest.TestCase):
    def test_coordinate_stitcher_places_tiles_from_metadata(self):
        left = np.zeros((40, 60, 3), dtype=np.uint8)
        right = np.zeros((40, 60, 3), dtype=np.uint8)
        left[:, :, 1] = 100
        right[:, :, 2] = 180
        tiles = [
            TileRecord(row=0, col=0, x=0, y=0, z=0, filename="left.png", focus_score=1.0),
            TileRecord(row=0, col=1, x=40, y=0, z=0, filename="right.png", focus_score=1.0),
        ]

        result = stitch_tiles_by_stage_coordinates([left, right], tiles, pixels_per_pulse=1.0)

        self.assertEqual(result.shape, (40, 100, 3))
        self.assertEqual(int(result[10, 10, 1]), 100)
        self.assertEqual(int(result[10, 90, 2]), 180)

    def test_overlap_registration_corrects_neighbor_stage_error(self):
        rng = np.random.default_rng(7)
        base = rng.integers(0, 255, size=(40, 80, 3), dtype=np.uint8)
        left = base[:, 0:50].copy()
        right = base[:, 30:80].copy()
        tiles = [
            TileRecord(row=0, col=0, x=0, y=0, z=0, filename="left.png", focus_score=1.0),
            TileRecord(row=0, col=1, x=35, y=0, z=0, filename="right.png", focus_score=1.0),
        ]

        positions = refine_tile_positions_by_overlap([left, right], tiles, pixels_per_pulse=1.0, max_correction=10)

        self.assertEqual(positions[0], (0, 0))
        self.assertEqual(positions[1], (30, 0))

    def test_registered_stitcher_uses_overlap_corrected_positions(self):
        rng = np.random.default_rng(8)
        base = rng.integers(0, 255, size=(40, 80, 3), dtype=np.uint8)
        left = base[:, 0:50].copy()
        right = base[:, 30:80].copy()
        tiles = [
            TileRecord(row=0, col=0, x=0, y=0, z=0, filename="left.png", focus_score=1.0),
            TileRecord(row=0, col=1, x=35, y=0, z=0, filename="right.png", focus_score=1.0),
        ]

        result = stitch_tiles_by_stage_coordinates(
            [left, right],
            tiles,
            pixels_per_pulse=1.0,
            use_overlap_registration=True,
        )

        self.assertEqual(result.shape, (40, 80, 3))

    def test_stitch_session_by_metadata_writes_output_image(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = np.zeros((20, 30, 3), dtype=np.uint8)
            right = np.zeros((20, 30, 3), dtype=np.uint8)
            left[:, :, 0] = 80
            right[:, :, 2] = 160
            cv2.imwrite(str(root / "left.png"), left)
            cv2.imwrite(str(root / "right.png"), right)
            metadata = {
                "tiles": [
                    {"row": 0, "col": 0, "x": 0, "y": 0, "z": 0, "filename": "left.png", "focus_score": 1.0},
                    {"row": 0, "col": 1, "x": 20, "y": 0, "z": 0, "filename": "right.png", "focus_score": 1.0},
                ]
            }
            (root / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")

            output = stitch_session_by_metadata(root, pixels_per_pulse=1.0)

            self.assertTrue(output.exists())
            mosaic = cv2.imread(str(output))
            self.assertEqual(mosaic.shape, (20, 50, 3))


if __name__ == "__main__":
    unittest.main()
