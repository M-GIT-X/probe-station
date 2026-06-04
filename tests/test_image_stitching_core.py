import json
import unittest

import numpy as np

from sample_plane import (
    SamplePlanePoint,
    boundary_polygon_from_points,
    evaluate_plane_consistency,
    fit_sample_plane,
    fit_sample_plane_robust,
    generate_plane_probe_points,
)
from scan_plan import ScanBounds, generate_overlap_scan_plan, generate_stitching_grid
from stitching_store import StitchingSessionStore, TileRecord
from stitching_calibration import calibration_from_shifts


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

    def test_plane_consistency_accepts_four_coplanar_focused_corners(self):
        corners = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c2", 100, 0, 110),
            SamplePlanePoint("c3", 100, 100, 130),
            SamplePlanePoint("c4", 0, 100, 120),
        ]

        report = evaluate_plane_consistency(corners, max_confirmation_residual=5.0)

        self.assertTrue(report.accepted)
        self.assertIsNone(report.suspicious_label)
        self.assertLessEqual(report.max_confirmation_residual, 1e-6)

    def test_plane_consistency_rejects_four_corners_that_do_not_confirm_each_other(self):
        corners = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c2", 100, 0, 110),
            SamplePlanePoint("c3", 100, 100, 180),
            SamplePlanePoint("c4", 0, 100, 120),
        ]

        report = evaluate_plane_consistency(corners, max_confirmation_residual=20.0)

        self.assertFalse(report.accepted)
        self.assertGreater(report.max_confirmation_residual, 20.0)
        self.assertIn("confirmation residual", report.message)

    def test_generate_plane_probe_points_samples_three_by_three_inside_boundary(self):
        corners = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c2", 100, 0, 110),
            SamplePlanePoint("c3", 100, 100, 130),
            SamplePlanePoint("c4", 0, 100, 120),
        ]
        plane = fit_sample_plane(corners)

        probes = generate_plane_probe_points(corners, plane, grid_size=3)

        self.assertEqual(len(probes), 9)
        self.assertEqual((probes[0].x, probes[0].y), (0, 0))
        self.assertEqual((probes[4].x, probes[4].y), (50, 50))
        self.assertEqual(probes[4].z, 115)

    def test_robust_plane_fit_drops_one_bad_autofocus_probe(self):
        points = [
            SamplePlanePoint("p1", 0, 0, 100),
            SamplePlanePoint("p2", 50, 0, 105),
            SamplePlanePoint("p3", 100, 0, 110),
            SamplePlanePoint("p4", 0, 50, 110),
            SamplePlanePoint("p5", 50, 50, 200),
            SamplePlanePoint("p6", 100, 50, 120),
            SamplePlanePoint("p7", 0, 100, 120),
            SamplePlanePoint("p8", 50, 100, 125),
            SamplePlanePoint("p9", 100, 100, 130),
        ]

        result = fit_sample_plane_robust(points, max_abs_residual=8.0, min_inliers=6)

        self.assertEqual([point.label for point in result.outliers], ["p5"])
        self.assertEqual(result.plane.z_at(50, 50), 115)
        self.assertLessEqual(result.plane.max_abs_residual, 1e-6)

    def test_boundary_polygon_orders_corners_for_safe_in_bounds_checks(self):
        unordered = [
            SamplePlanePoint("c1", 0, 0, 100),
            SamplePlanePoint("c3", 100, 100, 100),
            SamplePlanePoint("c2", 100, 0, 100),
            SamplePlanePoint("c4", 0, 100, 100),
        ]

        boundary = boundary_polygon_from_points(unordered)

        self.assertEqual(set(boundary), {(0, 0), (100, 0), (100, 100), (0, 100)})
        self.assertNotEqual(boundary, [(0, 0), (100, 100), (100, 0), (0, 100)])

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

    def test_overlap_scan_plan_derives_tile_count_and_snake_grid_from_calibration(self):
        plane = fit_sample_plane(
            [
                SamplePlanePoint("c1", 0, 0, 100),
                SamplePlanePoint("c2", 1000, 0, 100),
                SamplePlanePoint("c3", 1000, 600, 120),
                SamplePlanePoint("c4", 0, 600, 120),
            ]
        )
        bounds = ScanBounds(min_x=0, max_x=1000, min_y=0, max_y=600)

        plan = generate_overlap_scan_plan(
            bounds,
            plane,
            frame_width=400,
            frame_height=300,
            x_pixels_per_pulse=0.5,
            y_pixels_per_pulse=0.5,
            overlap_percent=25.0,
        )

        self.assertEqual(plan.cols, 3)
        self.assertEqual(plan.rows, 3)
        self.assertEqual(len(plan.tiles), 9)
        self.assertEqual([(tile.row, tile.col) for tile in plan.tiles[3:6]], [(1, 2), (1, 1), (1, 0)])
        self.assertLessEqual(plan.x_step_pulses, 600)
        self.assertLessEqual(plan.y_step_pulses, 450)

    def test_overlap_scan_plan_drops_centers_outside_four_corner_polygon(self):
        plane = fit_sample_plane(
            [
                SamplePlanePoint("c1", 50, 0, 100),
                SamplePlanePoint("c2", 100, 50, 100),
                SamplePlanePoint("c3", 50, 100, 100),
                SamplePlanePoint("c4", 0, 50, 100),
            ]
        )
        bounds = ScanBounds(min_x=0, max_x=100, min_y=0, max_y=100)
        boundary = [(50, 0), (100, 50), (50, 100), (0, 50)]

        plan = generate_overlap_scan_plan(
            bounds,
            plane,
            frame_width=50,
            frame_height=50,
            x_pixels_per_pulse=1.0,
            y_pixels_per_pulse=1.0,
            overlap_percent=20.0,
            boundary_points=boundary,
        )

        self.assertTrue(plan.tiles)
        self.assertNotIn((0, 0), [(tile.x, tile.y) for tile in plan.tiles])

    def test_overlap_scan_plan_does_not_cap_user_requested_large_range(self):
        plane = fit_sample_plane(
            [
                SamplePlanePoint("c1", 0, 0, 100),
                SamplePlanePoint("c2", 200_000, 0, 100),
                SamplePlanePoint("c3", 200_000, 120_000, 100),
                SamplePlanePoint("c4", 0, 120_000, 100),
            ]
        )
        bounds = ScanBounds(min_x=0, max_x=200_000, min_y=0, max_y=120_000)

        plan = generate_overlap_scan_plan(
            bounds,
            plane,
            frame_width=1000,
            frame_height=800,
            x_pixels_per_pulse=1.0,
            y_pixels_per_pulse=1.0,
            overlap_percent=20.0,
        )

        self.assertEqual(plan.cols, 251)
        self.assertEqual(plan.rows, 189)
        self.assertEqual(len(plan.tiles), 47_439)
        self.assertEqual((plan.tiles[0].x, plan.tiles[0].y), (0, 0))
        self.assertEqual((plan.tiles[-1].x, plan.tiles[-1].y), (200_000, 120_000))

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
            record = TileRecord(
                row=0,
                col=0,
                x=10,
                y=20,
                z=30,
                filename="",
                focus_score=12.5,
                mean_brightness=127.0,
                saturation_fraction=0.0,
                overexposed=False,
                sample_frames=3,
                saved_from_raw_frame=True,
            )

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
            self.assertEqual(metadata["tiles"][0]["mean_brightness"], 127.0)
            self.assertEqual(metadata["tiles"][0]["sample_frames"], 3)
            self.assertTrue(metadata["tiles"][0]["saved_from_raw_frame"])
            self.assertEqual(metadata["corners"][0]["label"], "c1")

    def test_stitching_store_writes_tile_quality_csv(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            store = StitchingSessionStore.create(Path(temp_dir), "quality")
            tiles = [
                TileRecord(
                    row=0,
                    col=1,
                    x=10,
                    y=20,
                    z=30,
                    filename="tile.png",
                    focus_score=12.5,
                    mean_brightness=127.0,
                    saturation_fraction=0.01,
                    underexposed_fraction=0.02,
                    overexposed=False,
                    sample_frames=5,
                    saved_from_raw_frame=True,
                )
            ]

            output = store.write_tile_quality_csv(tiles)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(output.name, "tile_quality.csv")
            self.assertEqual(rows[0]["filename"], "tile.png")
            self.assertEqual(rows[0]["focus_score"], "12.5")
            self.assertEqual(rows[0]["saved_from_raw_frame"], "True")

    def test_stitching_store_persists_automatic_calibration(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            store = StitchingSessionStore.create(Path(temp_dir), "calibrated")
            calibration = calibration_from_shifts(
                x_step_pulses=10,
                y_step_pulses=10,
                x_frame_shift=(-5.0, 0.0),
                y_frame_shift=(0.0, -4.0),
                frame_width=100,
                frame_height=80,
                overlap_percent=20.0,
                confidence=0.8,
            )

            store.write_metadata(corners=[], tiles=[], settings={}, calibration=calibration)

            metadata = json.loads((store.path / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["calibration"]["x_pixels_per_pulse"], 0.5)
            self.assertEqual(metadata["calibration"]["overlap_percent"], 20.0)

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
