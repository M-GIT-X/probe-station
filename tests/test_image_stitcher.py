import tempfile
from pathlib import Path
import unittest

import numpy as np

import image_stitcher
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

    def test_automatic_mosaic_does_not_amplify_fixed_camera_shading(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shading = np.full((30, 50, 3), 80, dtype=np.uint8)
            shading[:, 20:, :] = 180
            cv2.imwrite(str(root / "left.png"), shading)
            cv2.imwrite(str(root / "right.png"), shading)
            metadata = {
                "tiles": [
                    {"row": 0, "col": 0, "x": 0, "y": 0, "z": 0, "filename": "left.png", "focus_score": 1.0},
                    {"row": 0, "col": 1, "x": 30, "y": 0, "z": 0, "filename": "right.png", "focus_score": 1.0},
                ]
            }
            (root / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")

            output = stitch_session_by_metadata(root, pixels_per_pulse=1.0, use_overlap_registration=False)
            mosaic = cv2.imread(str(output))

        self.assertLessEqual(int(mosaic.max()), 180)

    def test_overlap_blending_feathers_different_tile_brightness_across_join(self):
        left = np.full((30, 50, 3), 120, dtype=np.uint8)
        right = np.full((30, 50, 3), 60, dtype=np.uint8)
        tiles = [
            TileRecord(row=0, col=0, x=0, y=0, z=0, filename="left.png", focus_score=1.0),
            TileRecord(row=0, col=1, x=30, y=0, z=0, filename="right.png", focus_score=1.0),
        ]

        mosaic = stitch_tiles_by_stage_coordinates([left, right], tiles, pixels_per_pulse=1.0)
        overlap_values = np.unique(mosaic[15, 30:50, 0])

        self.assertGreater(len(overlap_values), 3)

    def test_overlap_registration_limits_full_search_work_for_camera_sized_frames(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")

        calls = []
        original_overlap_mse = image_stitcher._overlap_mse

        def counted_overlap_mse(anchor, moving, dx, dy):
            calls.append((dx, dy))
            return float((dx - 90) ** 2 + dy**2)

        try:
            image_stitcher._overlap_mse = counted_overlap_mse
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            image_stitcher._estimate_neighbor_delta(
                frame,
                frame,
                expected_delta=(90, 0),
                max_correction=30,
            )
        finally:
            image_stitcher._overlap_mse = original_overlap_mse

        self.assertLess(len(calls), 1000)

    def test_multiscale_overlap_registration_preserves_large_frame_alignment_accuracy(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")

        rng = np.random.default_rng(17)
        base = rng.integers(0, 255, size=(480, 1000, 3), dtype=np.uint8)
        left = base[:, 0:640].copy()
        right = base[:, 300:940].copy()

        delta = image_stitcher._estimate_neighbor_delta(
            left,
            right,
            expected_delta=(315, 0),
            max_correction=60,
        )

        self.assertEqual(delta, (300, 0))

    def test_signed_axis_calibration_places_tiles_in_camera_orientation(self):
        left = np.zeros((20, 30, 3), dtype=np.uint8)
        right = np.zeros((20, 30, 3), dtype=np.uint8)
        left[:, :, 1] = 80
        right[:, :, 2] = 160
        tiles = [
            TileRecord(row=0, col=0, x=0, y=0, z=0, filename="left.png", focus_score=1.0),
            TileRecord(row=0, col=1, x=20, y=0, z=0, filename="right.png", focus_score=1.0),
        ]

        mosaic = stitch_tiles_by_stage_coordinates(
            [left, right],
            tiles,
            x_pixels_per_pulse=-1.0,
            y_pixels_per_pulse=1.0,
        )

        self.assertEqual(int(mosaic[5, 5, 2]), 160)
        self.assertEqual(int(mosaic[5, 45, 1]), 80)

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

    def test_stitch_session_reads_automatic_axis_calibration_from_metadata(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "left.png"), frame)
            cv2.imwrite(str(root / "right.png"), frame)
            metadata = {
                "calibration": {"x_pixels_per_pulse": 0.5, "y_pixels_per_pulse": 0.5},
                "tiles": [
                    {"row": 0, "col": 0, "x": 0, "y": 0, "z": 0, "filename": "left.png", "focus_score": 1.0},
                    {"row": 0, "col": 1, "x": 40, "y": 0, "z": 0, "filename": "right.png", "focus_score": 1.0},
                ],
            }
            (root / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")

            output = stitch_session_by_metadata(root, use_overlap_registration=False)

            mosaic = cv2.imread(str(output))
            self.assertEqual(mosaic.shape, (20, 50, 3))


if __name__ == "__main__":
    unittest.main()
