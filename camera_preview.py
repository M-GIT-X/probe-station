"""Show a local camera feed and the V2 sharpness score for each displayed frame."""

from __future__ import annotations

import argparse
import math
import sys

from autofocus_v2.capture import CaptureError, FrameStream
from autofocus_v2.metrics import evaluate_frame
from autofocus_v2.models import ROI
from camera_opencv import BACKENDS, cv2


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
    parser.add_argument("--exposure", type=float, help="override the camera's startup exposure")
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
    try:
        root.title(f"Probe station camera {args.camera_index}")
        root.protocol("WM_DELETE_WINDOW", root.quit)
        for key in ("<q>", "<Q>", "<Escape>"):
            root.bind(key, lambda _event: root.quit())
        status = tk.Label(root, text="Opening camera…")
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
                    scored_roi = roi if roi is not None else ROI(0, 0, width, height)
                    measured = evaluate_frame(frame.image, scored_roi)
                    valid_tiles = sum(tile.valid for tile in measured.tiles)
                    score = f"{measured.score:.4g}" if measured.valid else "invalid exposure"
                    status.configure(text=f"V2 score: {score}  ({valid_tiles}/4 tiles)")

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

        stream.start(args.camera_index, args.backend, settings)
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
