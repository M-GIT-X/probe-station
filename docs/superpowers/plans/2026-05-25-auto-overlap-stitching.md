# Automatic Overlap Stitching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator specify overlap only, while the application safely calibrates image scale inside the four recorded corners, plans the scan automatically, and stitches tiles using overlap registration.

**Architecture:** Add a small calibration module that converts in-memory trial-frame translations into signed X/Y pixel-per-pulse values. Extend scan planning to derive rows, columns, and tile centers from calibrated camera coverage and requested overlap; the GUI worker performs three safe in-bound trial captures before planning and scanning.

**Tech Stack:** Python, NumPy, OpenCV phase correlation, Tkinter, unittest.

---

### Task 1: Calibration Model And Automatic Scan Plan

**Files:**
- Create: `stitching_calibration.py`
- Modify: `scan_plan.py`
- Modify: `pyproject.toml`
- Test: `tests/test_stitching_calibration.py`
- Test: `tests/test_image_stitching_core.py`

- [ ] Write tests proving trial points stay inside `ScanBounds`, signed pixel-per-pulse is computed from image shifts, and a 25% overlap automatically creates a snake plan from frame size and fitted plane.
- [ ] Run `uv run python -m unittest tests.test_stitching_calibration tests.test_image_stitching_core -v` and confirm imports/functions fail before implementation.
- [ ] Implement `StitchingCalibration`, in-bound trial point selection, shift-based calibration, and `generate_overlap_scan_plan`.
- [ ] Run the same test command and confirm it passes.

### Task 2: Calibrated Stitching Placement

**Files:**
- Modify: `image_stitcher.py`
- Modify: `stitching_store.py`
- Test: `tests/test_image_stitcher.py`

- [ ] Write tests proving separate signed X/Y scales determine placement and calibration metadata is used when rebuilding a session mosaic.
- [ ] Run `uv run python -m unittest tests.test_image_stitcher -v` and confirm the new expectations fail.
- [ ] Accept calibrated X/Y placement in the stitcher while retaining overlap refinement and low-texture fallback; persist calibration in metadata.
- [ ] Run the image stitcher tests and confirm they pass.

### Task 3: Safe GUI Orchestration

**Files:**
- Modify: `gui_app.py`
- Modify: `README.md`
- Test: `tests/test_app_modes.py`

- [ ] Write/update tests that define the reduced stitching controls: overlap is user-facing; automatically calculated scan/scale fields are status outputs.
- [ ] Run `uv run python -m unittest tests.test_app_modes -v` and confirm the new assertions fail.
- [ ] Replace manual rows/columns/pixel-scale inputs with overlap; in the stitching worker move to an interior reference point, capture reference/X/Y trial frames after arrival and settling, reject failed calibration, generate tiles, display the calculated plan, then scan and stitch.
- [ ] Document the calibrated workflow and run `uv run python -m unittest discover -s tests -v` plus `uv run python -m compileall gui_app.py image_stitcher.py scan_plan.py stitching_calibration.py stitching_store.py`.
