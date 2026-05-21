import tempfile
from pathlib import Path
import unittest

import numpy as np

from image_stitcher import stitch_session_by_metadata, stitch_tiles_by_stage_coordinates
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
