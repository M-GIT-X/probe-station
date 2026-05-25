# Image Stitching MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first image-stitching workflow where the operator records four manually focused corners, the app fits a sample plane, scans tiles with Z compensation, saves images plus metadata, and can run an offline OpenCV stitch.

**Architecture:** Keep stitching logic out of the Tkinter app. Add small modules for sample-plane fitting, grid planning, tile storage, and image stitching; the GUI only records corners, starts/stops the scan, and displays progress/output paths.

**Tech Stack:** Python 3.10+, Tkinter, NumPy, OpenCV, pyserial, unittest/pytest-compatible tests.

---

### Task 1: Core Geometry And Metadata

**Files:**
- Create: `sample_plane.py`
- Create: `scan_plan.py`
- Create: `stitching_store.py`
- Test: `tests/test_image_stitching_core.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing tests**

```python
def test_plane_fit_predicts_z_inside_four_focused_corners():
    corners = [
        SamplePlanePoint("c1", 0, 0, 100),
        SamplePlanePoint("c2", 100, 0, 110),
        SamplePlanePoint("c3", 100, 100, 130),
        SamplePlanePoint("c4", 0, 100, 120),
    ]
    plane = fit_sample_plane(corners)
    assert round(plane.z_at(50, 50)) == 115
    assert plane.max_abs_residual <= 1e-6
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_image_stitching_core.py -q`
Expected: import failure because the new modules do not exist yet.

- [ ] **Step 3: Implement minimal core modules**

Create dataclasses for plane points, fitted plane, tile points, and tile metadata. Generate a snake-path grid over the rectangular min/max X/Y bounds.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_image_stitching_core.py -q`
Expected: all core tests pass.

### Task 2: Offline Image Stitching

**Files:**
- Create: `image_stitcher.py`
- Test: `tests/test_image_stitcher.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing tests**

```python
def test_coordinate_stitcher_places_tiles_from_metadata():
    base = np.zeros((40, 60, 3), dtype=np.uint8)
    base[:, :, 1] = 100
    tiles = [
        TileRecord(row=0, col=0, x=0, y=0, z=0, filename="a.png", focus_score=1.0),
        TileRecord(row=0, col=1, x=40, y=0, z=0, filename="b.png", focus_score=1.0),
    ]
    result = stitch_tiles_by_stage_coordinates([base, base], tiles, pixels_per_pulse=1.0)
    assert result.shape == (40, 100, 3)
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_image_stitcher.py -q`
Expected: import failure because `image_stitcher.py` does not exist.

- [ ] **Step 3: Implement minimal coordinate stitcher**

Use stage metadata for deterministic placement, with an optional OpenCV phase-correlation helper for later refinement.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_image_stitcher.py -q`
Expected: image stitcher tests pass.

### Task 3: GUI Integration

**Files:**
- Modify: `gui_app.py`
- Test: `tests/test_app_modes.py`

- [ ] **Step 1: Write failing tests**

Add assertions that `Mode.IMAGE_STITCHING` exists and its mode panel exposes stitching actions/status fields.

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_app_modes.py -q`
Expected: enum value or panel assertion failure.

- [ ] **Step 3: Add the GUI panel**

Add an Image Stitching mode with controls for recording/deleting corners, configuring rows/cols/overlap/settle/sample count/output folder, starting/stopping scan, and running offline stitch.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_app_modes.py -q`
Expected: app mode tests pass.

### Task 4: End-To-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the operator workflow**

Describe four-corner recording, safety assumptions, output folder contents, and offline stitch behavior.

- [ ] **Step 2: Run full tests**

Run: `uv run pytest -q`
Expected: all tests pass.
