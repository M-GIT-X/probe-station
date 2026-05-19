# Three-Axis Probe Station

Clean first-pass Python project for a Windows lab computer controlling a three-axis X/Y/Z probe station.

## Install

```bat
cd probe_station
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bat
python main.py
```

The main window appears first. It does not scan cameras or open the motor controller at startup.
Runtime errors are written to `debug.log` beside `main.py`.

## First Real-Hardware Test

1. Verify that a physical emergency stop is installed, reachable, and tested before enabling motion.
2. Start the GUI with no devices connected and confirm the window opens quickly.
3. Connect only the motor controller, enter the serial port such as `COM5`, and click `Connect Motor`.
4. Use a very small step such as `10` to `100` pulses and a low speed such as `5` to `10`.
5. Test `X+`, `X-`, `Y+`, `Y-`, `Z+`, and `Z-` one at a time.
6. Confirm that GUI `X+` and `X-` match the desired lab-coordinate direction. The controller direction for X is intentionally inverted in software.
7. Test `Space` for controlled stop.
8. Test `E` or `Esc` for software emergency stop while keeping the physical emergency stop ready.
9. Open the camera only after motor startup is stable. If the camera cannot open, the GUI remains in no-camera mode.
10. Start `Manual Focus Assist`, move Z manually, and use `Go To Best Z` only after confirming Z motion direction and step size.

## Safety Notes

- This first version is intentionally conservative: no automatic scan autofocus is implemented yet.
- The stage has only X/Y/Z in this software. Controller A-axis support is ignored.
- Do not use hardware home command `D0`; there are currently no limit or home sensors.
- `Set Current As Software Origin` only changes local relative coordinates. It does not send `D3` software clear to the controller.
- Software emergency stop is retained, but it cannot replace a physical emergency stop.
- Keep default motion low-speed and small-step for first tests.

## Module Layout

- `stage_protocol.py`: frame building, checksum, parsing, and protocol command helpers for X/Y/Z/ALL only.
- `stage_controller.py`: pyserial connection, reads, manual relative movement, controlled stop, and software emergency stop.
- `camera_opencv.py`: OpenCV camera open/close/read only; no startup camera scan.
- `focus_metrics.py`: ROI selection and robust focus index calculation.
- `app_gui.py`: Tkinter GUI, keyboard shortcuts, camera display via PPM `PhotoImage`, manual focus assist.
- `main.py`: logging and GUI startup only.

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
- `E` and `Esc` send `FC FF 49...` for all-axis software emergency stop.
