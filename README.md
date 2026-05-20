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

## Modes

Use the GUI `Mode Selector` to switch between:

- `Manual Mode`: manual X/Y/Z movement only. Focus index is shown, but no best
  focus is recorded and no automatic movement happens.
- `Manual Focus Assist`: the user still moves Z manually. The GUI records focus
  index and best Z, then `Go To Best Z` can return to the recorded Z at low speed
  after confirmation.
- `Auto Focus`: conservative full-scan autofocus. It only moves Z. It does not
  move X/Y, perform snake scanning, stitching, or multi-point measurement.

## Shortcuts

- `A` / `D` = X- / X+
- `W` / `S` = Y+ / Y-
- `Q` / `E` = Z- / Z+
- `Space` = all-axis decelerated stop
- `Esc` = software emergency stop

`E` is Z+ only. Software emergency stop is `Esc` or the GUI emergency-stop
button. 软件急停不能替代物理急停。

## Real-Hardware Test Order

1. Start in `Manual Mode` and test X/Y/Z with small steps.
2. Use `Manual Focus Assist` to verify focus index, best focus recording, and
   `Go To Best Z`.
3. Use `Auto Focus` only after manual movement and focus assist are stable.
4. Before autofocus, manually move close to focus.
5. First autofocus parameters:
   - `scan_range = 20`
   - `scan_step = 5`
   - `speed = 1` or `2`
   - `settle_seconds = 0.5`
   - `sample_seconds = 1.5`

## Camera Overexposure

If OpenCV looks much brighter than the Windows Camera app:

1. Open `Camera Controls`.
2. Click `Read Camera Properties`.
3. Click `Reduce Overexposure`.
4. If still too bright, turn off `Auto Exposure`, set `exposure = -6`, `-7`, or
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
- Press `Space`, `Esc`, or the physical emergency stop immediately if anything
  looks unsafe.

`SAFE_MODE` is enabled in `gui_app.py`:

- `SAFE_MAX_MANUAL_STEP = 50`
- `SAFE_MAX_MANUAL_SPEED = 5`
- `SAFE_MAX_AUTOFOCUS_RANGE = 100`
- `SAFE_MAX_AUTOFOCUS_STEP = 20`
- `SAFE_MAX_AUTOFOCUS_SPEED = 5`

Inputs above those limits are clamped and logged.

## Direction Mapping

Direction constants are in `gui_app.py`:

```python
INVERT_X_DIRECTION = True
INVERT_Y_DIRECTION = True
INVERT_Z_DIRECTION = False
```

X and Y are software-inverted based on real-machine testing. Z keeps the
successful Manual Focus Assist direction.

## Module Layout

- `main.py`: single recommended entry point.
- `gui_app.py`: unified GUI, mode selector, manual controls, camera controls,
  Manual Focus Assist, and Auto Focus.
- `stage_protocol.py`: 12-byte protocol frame helpers for X/Y/Z/ALL only.
- `stage_controller.py`: serial connection, D4 realtime-upload disable on open,
  movement, B5 arrival wait, position reads, stop, and emergency stop.
- `camera_opencv.py`: OpenCV camera open/close/read, backend selection, property
  controls, and `reduce_overexposure()`.
- `focus_metrics.py`: automatic ROI selection, relative focus index, robust
  representative score, and brightness/saturation diagnostics.
- `app_gui.py`: compatibility shim importing from `gui_app.py`.

## Auto Focus Function

`ProbeStationApp._autofocus_worker()` performs the conservative full scan:

1. sample baseline at offset 0;
2. move Z to `-scan_range`;
3. scan offsets through `+scan_range`;
4. sample focus index at each point;
5. choose a near-best stable point;
6. move Z to the final offset;
7. confirm focus score.

`Stop Autofocus` and `Space` set the AF stop flag and call `stage.stop_all()`.
`Esc` sets the same stop flag and calls `stage.emergency_stop_all()`.

## Logging

`debug.log` records startup, motor connection, camera connection, camera
property reads/writes, Reduce Overexposure steps, mode switches, manual moves,
Stop, Emergency Stop, Manual Focus Assist events, autofocus points/final offset,
confirm score, and exception tracebacks.
