# Three-Axis Probe Station

Unified Tkinter GUI for a lab X/Y/Z automated probe station. The old
`main_stage1_manual.py`, `main_stage2_focus_assist.py`, and
`main_stage3_autofocus.py` files are retained only as legacy forwarders. Use
`main.py` for real operation.

## Run

```bat
python main.py
```

or:

```bat
uv run python main.py
```

The app starts with one visible main window. It does not scan cameras or connect
to the motor controller at startup. Runtime diagnostics are written to
`debug.log` beside `main.py`.

## Windows COM Port Troubleshooting

Check which ports Windows currently sees:

```bat
uv run python -m serial.tools.list_ports -v
```

If `COM5` is listed but opening it fails with
`PermissionError(13, 'A device attached to the system is not functioning.', ..., 31)`,
the GUI now retries with Windows-specific serial open fallbacks:

- the normal `COM5` name;
- the Windows device path form `\\.\COM5`;
- versions without `write_timeout`;
- versions that keep RTS/DTR low before opening the port.

This keeps normal Windows behavior unchanged when the standard open works, but
helps USB-serial drivers that fail during Windows `SetCommState` configuration.
If all attempts fail, disable/enable the COM port in Device Manager or reinstall
the USB-serial driver.

## Modes

Use the top mode drop-down to switch between:

- `Manual Mode`: manual X/Y/Z movement only. Sharpness index is shown, but no best
  focus is recorded and no automatic movement happens.
- `Auto Focus`: conservative full-scan autofocus. It only moves Z. It does not
  move X/Y, perform snake scanning, stitching, or multi-point measurement.
  `Semi Auto` uses the user range/step. `Full Auto` currently accepts only range
  and automatically runs coarse-to-fine Z scans around the best point.
- `Image Stitching`: four-corner stitching workflow. The user manually moves to
  four focused corners, records each X/Y/Z point, then the GUI fits a sample
  plane. On start, the GUI performs in-bound trial captures to calibrate image
  scale, derives the tile grid from the requested overlap, and scans with Z
  compensation. Captured tiles and `metadata.json` are saved in a timestamped
  folder, followed by `stitched_mosaic.png`.

## Shortcuts

- `A` / `D` = X+ / X-
- `W` / `S` = Y- / Y+
- `Q` / `E` = Z- / Z+
- `Space` = all-axis decelerated stop
- `Esc` = software emergency stop

`E` is Z+ only. Software emergency stop is `Esc` or the GUI emergency-stop
button. 软件急停不能替代物理急停。

Press `Enter` after editing numeric fields such as step, speed, scan range, or
camera exposure to return focus to the main window before using movement
shortcuts.

## Real-Hardware Test Order

1. Start in `Manual Mode` and test X/Y/Z with small steps.
2. Use `Auto Focus` only after manual movement and live sharpness readings are stable.
3. Before autofocus, manually move close to focus.
4. First autofocus parameters:
   - `half-range = 20`
   - `scan_step = 5`
   - `speed = 1` or `2`
   - `settle_seconds = 0.5`
   - `sample_seconds = 1.5`
5. Use `Image Stitching` only after manual X/Y/Z and camera capture are stable.
   For the first scan, record a small four-corner safe area with no probe
   contact; the application calculates the tile count automatically.

## Image Stitching Workflow

1. Connect the stage and open the camera.
2. Switch to `Image Stitching`.
3. Manually move to the first corner of the desired scan area.
4. Manually focus at that corner, then click `Record Corner`.
5. Repeat for four corners around the desired scan area.
6. Check the displayed plane residual. A large residual means the corner focus
   points disagree; clear or delete corners and record them again.
7. Choose `Overlap %` plus conservative motion/capture settings (`Speed %`,
   `Settle s`, and `Frames/tile`).
8. Click `Start Stitching Scan` and confirm the safety dialog.
9. Before scanning, the GUI moves to three in-bound trial positions, captures
   frames in memory, estimates X/Y pixels per pulse, and automatically plans
   the snake path. Trial images are not saved as tile files.
10. The GUI saves tile images and `metadata.json` in a new timestamped session
    folder under `stitching_output/`. Metadata includes automatic calibration.
11. The scan automatically stitches the captured tiles, writes
    `stitched_mosaic.png`, and displays a completion notification with its
    saved path.

The stitcher starts from stage coordinates, then tries to refine adjacent tile
placement by matching their overlapping image regions. If the overlap has too
little texture for reliable matching, it falls back to the stage-coordinate
placement instead of forcing a bad correction. Each tile is sampled multiple
times and the clearest acceptable frame is saved, which helps while the
temporary setup is still vibration-prone.

Overlapping regions are feather blended when the mosaic is written. The
stitcher intentionally does not propagate per-tile brightness gains, because
fixed camera or illumination shading can otherwise become stronger stripes.

During automatic mosaic generation, neighboring tile overlaps are also used to
estimate gentle brightness compensation. This reduces alternating bright/dark
bands caused by capture exposure variation without independently normalizing
away real sample contrast.

The dynamic display below the shared camera preview outlines the planned scan
grid in green and fills each captured tile area as scanning progresses.

## Camera Overexposure

If OpenCV looks much brighter than the Windows Camera app:

1. Open `Camera Controls`.
2. On camera connection, the GUI automatically tests exposure values `-3` to `-11` and keeps
   the exposure with the best focus score while avoiding heavy saturation.
3. Click `Read Camera Properties`.
4. Click `Reduce Overexposure` if the automatic exposure tune is still too bright.
5. If still too bright, turn off `Auto Exposure`, set `exposure = -6`, `-7`, or
   `-8`, set `gain = 0`, then click `Apply Camera Settings`.

Before autofocus, target `saturation_fraction < 0.05`; at minimum keep it below
`0.10` when possible. If `saturation_fraction > 0.30`, autofocus shows a strong
warning and lets the user cancel or continue.

## Safety

- Software emergency stop cannot replace a physical emergency stop.
- The current system has no trusted limit/home hardware in this software path.
- The GUI does not use `D0` hardware home or `D3` controller clear.
- The project ignores the controller A axis; the real stage is X/Y/Z only.
- Auto Focus only moves Z.
- Image Stitching moves X/Y/Z automatically inside the recorded four-corner
  polygon. It requires exactly four recorded corners, keeps calibration trial
  locations inside the polygon, and does not impose a fixed tile-count limit.
- Press `Space`, `Esc`, or the physical emergency stop immediately if anything
  looks unsafe.

`SAFE_MODE` is enabled in `gui_app.py`:

- `SAFE_MAX_MANUAL_STEP = 2000`
- `SAFE_MAX_MANUAL_SPEED = 100`
- `SAFE_MAX_AUTOFOCUS_RANGE = 5000`
- `SAFE_MAX_AUTOFOCUS_STEP = 2000`
- `SAFE_MAX_AUTOFOCUS_SPEED = 100`

Inputs above those limits are clamped and logged.

## Direction Mapping

Direction constants are in `gui_app.py`:

```python
INVERT_X_DIRECTION = False
INVERT_Y_DIRECTION = True
INVERT_Z_DIRECTION = False
```

X was swapped after later real-machine manual-control feedback. Y remains
software-inverted. Z retains its verified manual-control direction.

## Module Layout

- `main.py`: single recommended entry point.
- `gui_app.py`: unified GUI, top mode selector, shared camera preview, manual
  controls, camera controls, Auto Focus, and Image Stitching.
- `stage_protocol.py`: 12-byte protocol frame helpers for X/Y/Z/ALL only.
- `stage_controller.py`: serial connection, D4 realtime-upload disable on open,
  movement, B5 arrival wait, position reads, stop, and emergency stop.
- `camera_opencv.py`: OpenCV camera open/close/read, backend selection, exposure
  selection helpers, property controls, and `reduce_overexposure()`.
- `focus_metrics.py`: automatic ROI selection, relative focus index, robust
  representative score, and brightness/saturation diagnostics.
- `sample_plane.py`: fits `Z = aX + bY + c` from manually focused corner points.
- `scan_plan.py`: generates snake-order image-stitching tile grids.
- `stitching_calibration.py`: estimates signed X/Y image scale using safe
  in-bound trial captures and the requested overlap.
- `stitching_store.py`: saves stitching tile images and session metadata.
- `image_stitcher.py`: offline coordinate-based mosaic generation, with a
  reserved OpenCV stitcher helper for future comparison.
- `app_gui.py`: compatibility shim importing from `gui_app.py`.

## Auto Focus Function

`ProbeStationApp._autofocus_worker()` performs the autofocus scan:

1. sample baseline at offset 0;
2. `half-range` means distance from the current Z to either scan edge; the total
   scanned width is approximately `2 * half-range`;
3. in `Semi Auto`, scan the user half-range/step once;
4. in `Full Auto`, scan coarse, then refine around the best point with smaller
   range and step;
5. sample focus index at each point;
6. choose a near-best stable point;
7. move Z to the final offset;
8. confirm focus score.

Before focus scoring, frames are translated against a reference frame using
phase correlation. This reduces table/camera jitter in the focus metric. It
cannot perfectly undo rolling-shutter S-shaped wobble, so the code also relies
on multi-frame robust scoring and IQR stability selection.

`Stop Autofocus` and `Space` set the AF stop flag and call `stage.stop_all()`.
`Esc` sets the same stop flag and calls `stage.emergency_stop_all()`.

## Logging

`debug.log` records startup, motor connection, camera connection, camera
property reads/writes, Reduce Overexposure steps, mode switches, manual moves,
Stop, Emergency Stop, autofocus points/final offset, confirm score, image
stitching progress, and exception tracebacks.
