"""Show a local camera feed and the V2 sharpness score for each displayed frame."""

from __future__ import annotations

import argparse
from collections import deque
import math
from statistics import mean, median
import sys

from autofocus_v2.capture import CaptureError, FrameStream
from autofocus_v2.metrics import evaluate_frame
from autofocus_v2.models import FrameScore, ROI
from camera_opencv import BACKENDS, cv2, exposure_tuning_candidates, startup_camera_settings


_STABILITY_FRAMES = 4
_RELATIVE_SPAN = 0.15


def _score_frame(image, roi: ROI | None) -> FrameScore:
    height, width = image.shape[:2]
    return evaluate_frame(image, roi if roi is not None else ROI(0, 0, width, height))


def _usable_score(frame: FrameScore) -> float | None:
    brightness = mean(tile.mean_brightness for tile in frame.tiles)
    saturation = mean(tile.saturation_fraction for tile in frame.tiles)
    dark = mean(tile.dark_fraction for tile in frame.tiles)
    if frame.valid and 5 < brightness < 230 and saturation <= 0.10 and dark <= 0.90:
        return frame.score
    return None


def _stable(scores) -> bool:
    if len(scores) < _STABILITY_FRAMES:
        return False
    center = median(scores)
    return center > 0 and max(scores) - min(scores) <= _RELATIVE_SPAN * center


def _exposure_scores(camera, stop, roi: ROI | None, exposure: float, reads: int):
    readback = camera.set_camera_property("exposure", exposure)
    if readback is None or not math.isclose(readback, exposure, abs_tol=0.1) or stop.wait(0.08):
        return
    # Drain a couple of reads after applying each candidate before scoring.
    for index in range(reads + 2):
        if stop.is_set():
            return
        image = camera.read_frame()
        if index >= 2:
            yield _usable_score(_score_frame(image, roi))


def _tune_exposure(camera, stop, roi: ROI | None, initial_exposure: float) -> str:
    candidates = []
    for exposure in exposure_tuning_candidates():
        if stop.is_set():
            return ""
        scores = list(_exposure_scores(camera, stop, roi, exposure, _STABILITY_FRAMES))
        usable = [score for score in scores if score is not None]
        if len(usable) == _STABILITY_FRAMES:
            candidates.append((_stable(usable), median(usable), exposure))

    if stop.is_set():
        return ""
    if not candidates:
        camera.set_camera_property("exposure", initial_exposure)
        return "Auto exposure unavailable. Check lighting or camera exposure control."

    _, _, exposure = max(candidates, key=lambda candidate: candidate[:2])
    recent = deque(maxlen=_STABILITY_FRAMES)
    for score in _exposure_scores(camera, stop, roi, exposure, reads=12):
        if score is None:
            recent.clear()
            continue
        recent.append(score)
        if _stable(recent):
            return f"Startup exposure {exposure:g} | V2 settled: {min(recent):.4g}–{max(recent):.4g}"
    return f"Startup exposure {exposure:g} | V2 stability unconfirmed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="local OpenCV camera index (default: 0)")
    parser.add_argument(
        "--backend", type=str.upper, choices=tuple(BACKENDS),
        default="DSHOW" if sys.platform == "win32" else "ANY",
        help="preferred OpenCV backend (default: DSHOW on Windows, ANY elsewhere)",
    )
    parser.add_argument(
        "--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="fixed ROI in original image pixels (default: entire frame)",
    )
    parser.add_argument("--exposure", type=float, help="use a fixed exposure instead of startup tuning")
    parser.add_argument("--gain", type=float, help="override the camera's startup gain")
    args = parser.parse_args(argv)
    if args.camera_index < 0:
        parser.error("--camera-index must be nonnegative")
    try:
        roi = ROI(*args.roi) if args.roi is not None else None
    except ValueError as error:
        parser.error(str(error))
    settings = {
        name: value for name, value in (("exposure", args.exposure), ("gain", args.gain))
        if value is not None
    }
    if any(not math.isfinite(value) for value in settings.values()):
        parser.error("--exposure and --gain must be finite")
    if cv2 is None:
        print("Camera preview requires opencv-python.", file=sys.stderr)
        return 1
    try:
        import tkinter as tk
    except ImportError:
        print("Camera preview requires Python with Tkinter support.", file=sys.stderr)
        return 1

    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Camera preview window could not open: {error}", file=sys.stderr)
        return 1

    stream = FrameStream()
    failure = None
    last_sequence = 0
    exposure_status = f"Fixed exposure requested: {args.exposure:g}" if args.exposure is not None else ""

    def calibrate(camera, stop) -> None:
        nonlocal exposure_status
        initial_exposure = float(startup_camera_settings(args.backend)["exposure"])
        exposure_status = _tune_exposure(camera, stop, roi, initial_exposure)

    try:
        root.title(f"Probe station camera {args.camera_index}")
        root.protocol("WM_DELETE_WINDOW", root.quit)
        for key in ("<q>", "<Q>", "<Escape>"):
            root.bind(key, lambda _event: root.quit())
        status = tk.Label(root, text=("Opening camera and tuning exposure… Keep the sample still."
                                      if args.exposure is None else "Opening camera…"))
        status.pack(pady=(6, 0))
        tk.Label(root, text="Press Q or Esc to quit, or close this window.").pack()
        preview = tk.Label(root)
        preview.pack(padx=8, pady=8)
        max_width = max(1, min(960, root.winfo_screenwidth() - 60))
        max_height = max(1, min(650, root.winfo_screenheight() - 150))

        def refresh() -> None:
            nonlocal failure, last_sequence
            try:
                camera_error = stream.error
                if camera_error is not None:
                    raise CaptureError(camera_error)
                frame = stream.latest()
                if frame is not None and frame.sequence != last_sequence:
                    height, width = frame.image.shape[:2]
                    measured = _score_frame(frame.image, roi)
                    valid_tiles = sum(tile.valid for tile in measured.tiles)
                    score = f"{measured.score:.4g}" if measured.valid else "invalid exposure"
                    status.configure(text=f"V2 score: {score}  ({valid_tiles}/4 tiles)\n{exposure_status}")

                    scale = min(1.0, max_width / width, max_height / height)
                    dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
                    small = cv2.resize(frame.image, (dw, dh), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB if small.ndim == 2 else cv2.COLOR_BGR2RGB)
                    if roi is not None:
                        left = roi.x * dw // width
                        top = roi.y * dh // height
                        right = (roi.x + roi.width - 1) * dw // width
                        bottom = (roi.y + roi.height - 1) * dh // height
                        cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 1)
                    ppm = f"P6\n{dw} {dh}\n255\n".encode() + rgb.tobytes()
                    photo = tk.PhotoImage(master=root, data=ppm, format="PPM")
                    preview.configure(image=photo)
                    preview.image = photo
                    last_sequence = frame.sequence
            except (CaptureError, ValueError, cv2.error, tk.TclError) as error:
                failure = str(error)
                root.quit()
                return
            root.after(30, refresh)

        stream.start(args.camera_index, args.backend, settings,
                     prepare=calibrate if args.exposure is None else None)
        root.after(0, refresh)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except tk.TclError as error:
        failure = str(error)
    finally:
        stream.stop(timeout=1.0)
        root.destroy()
    if failure is not None:
        print(f"Camera {args.camera_index} preview failed ({args.backend} preferred): {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
