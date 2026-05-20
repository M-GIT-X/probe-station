# Three-Axis Probe Station

Clean first-pass Python project for a Windows lab computer controlling a three-axis X/Y/Z probe station.

## Install With uv On Windows

This project is configured around `uv`, which creates and manages the Python
virtual environment from `pyproject.toml`.

### 1. Install uv

```bat
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen Command Prompt or PowerShell after installation so `uv` is on
`PATH`. Check it with:

```bat
uv --version
```

### 2. Create The Virtual Environment

From the project folder:

```bat
cd probe-station
uv sync
```

`uv sync` reads `.python-version` and `pyproject.toml`, then creates `.venv`
with Python 3.11 and installs:

- `pyserial` for the motor controller serial port
- `opencv-python` for the USB camera
- `numpy` for image and focus calculations

### 3. Run The App

Recommended:

```bat
uv run probe-station
```

Equivalent direct command:

```bat
uv run python main.py
```

The main window appears first. It does not scan cameras or open the motor controller at startup.
Runtime errors are written to `debug.log` beside `main.py`.

### 4. Run Stage 2 Manual Focus Assist

Stage 2 uses the same GUI code, with manual focus assist enabled and automatic autofocus disabled:

```bat
python main_stage2_focus_assist.py
```

If you are using `uv`, run:

```bat
uv run probe-station-stage2
```

## Updating Dependencies

After editing `pyproject.toml`, run:

```bat
uv sync
```

If you need a traditional activated shell for debugging:

```bat
.venv\Scripts\activate
python main.py
```

Deactivate it with:

```bat
deactivate
```

## Legacy pip Install

`requirements.txt` is kept for older setups, but new Windows computers should
use the uv workflow above.

## First Real-Hardware Test

1. Verify that a physical emergency stop is installed, reachable, and tested before enabling motion.
2. Start the GUI with no devices connected and confirm the window opens quickly.
3. Connect only the motor controller, enter the serial port such as `COM5`, and click `Connect Motor`.
4. Use a very small step such as `10` to `100` pulses and a low speed such as `5` to `10`.
5. Test `X+`, `X-`, `Y+`, `Y-`, `Z+`, and `Z-` one at a time.
6. Confirm that GUI `X+` and `X-` match the desired lab-coordinate direction. The controller direction for X is intentionally inverted in software.
7. Test `Space` for controlled stop.
8. Test `Esc` for software emergency stop while keeping the physical emergency stop ready.
9. Open the camera only after motor startup is stable. If the camera cannot open, the GUI remains in no-camera mode.
10. Start `Manual Focus Assist`, move Z manually, and use `Go To Best Z` only after confirming Z motion direction and step size.

## Keyboard Shortcuts

- `A` / `D` = X- / X+
- `W` / `S` = Y+ / Y-
- `Q` / `E` = Z- / Z+
- `R` / `F` = Z+ / Z- compatibility backup
- `Space` = all-axis decelerated stop
- `Esc` = software emergency stop
- `X` = backup software emergency stop

`E` is now Z+ only. It is not software emergency stop.
Axis movement shortcuts are ignored while the cursor is inside serial, camera,
step, or speed input fields. `Space` stop and `Esc` software emergency stop
remain available as safety controls.

## Stage 2 Manual Focus Assist Workflow

1. Run Stage 1 first and confirm that motor motion and the camera both work normally.
2. Run Stage 2 with `python main_stage2_focus_assist.py`.
3. Manually move to the target area.
4. Open the camera.
5. Click `Start Manual Focus Assist`.
6. Use `Q` / `E` with a small step size to move Z manually.
7. Watch `focus index`, current Z, and best focus values.
8. Click `Stop Manual Focus Assist` when you are done recording.
9. If needed, click `Go To Best Z`. Confirm the safety dialog before the GUI moves Z back to the recorded best absolute Z.
10. If anything looks wrong, immediately press `Esc` or use the physical emergency stop.

Stage 2 does not perform automatic focus scanning. It only records focus quality while the operator moves Z manually.

## Safety Notes

- This first version is intentionally conservative: no automatic scan autofocus is implemented yet.
- The stage has only X/Y/Z in this software. Controller A-axis support is ignored.
- Do not use hardware home command `D0`; there are currently no limit or home sensors.
- `Set Current As Software Origin` only changes local relative coordinates. It does not send `D3` software clear to the controller.
- Software emergency stop is retained, but it cannot replace a physical emergency stop. 软件急停不能替代物理急停。
- Keep default motion low-speed and small-step for first tests.

## Module Layout

- `stage_protocol.py`: frame building, checksum, parsing, and protocol command helpers for X/Y/Z/ALL only.
- `stage_controller.py`: pyserial connection, reads, manual relative movement, controlled stop, and software emergency stop.
- `camera_opencv.py`: OpenCV camera open/close/read only; no startup camera scan.
- `focus_metrics.py`: ROI selection and robust focus index calculation.
- `app_gui.py`: Tkinter GUI, keyboard shortcuts, camera display via PPM `PhotoImage`, manual focus assist.
- `main.py`: logging and GUI startup only.
- `main_stage2_focus_assist.py`: Stage 2 startup with `enable_focus_assist=True` and `enable_autofocus=False`.

## Controller Protocol

- Serial: `115200`, `8N1`, timeout `0.25 s`.
- Host command frames are fixed 12-byte frames: `3A function axis data[6] checksum 0D 0A`.
- Feedback frames are fixed 12-byte frames: `A3 function axis data[6] checksum 0D 0A`.
- The checksum is the low 8 bits of the sum of the first 9 bytes.
- Axis values: `X=0x01`, `Y=0x02`, `Z=0x04`, `ALL=0xFF`; A-axis is ignored.
- Opening the serial port sends `D4` once to disable realtime position upload.
- Relative movement uses `FA` with per-command speed percent.
- Movement completion waits for `B5` on the commanded axis before refreshing position.
- `Space` sends `FC FF 4A...` for all-axis decelerated stop.
- `Esc` and backup `X` send `FC FF 49...` for all-axis software emergency stop.
