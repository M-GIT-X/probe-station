"""Stage 1 entry point for manual motion and camera checks."""

from __future__ import annotations

import logging

from app_gui import ProbeStationApp
from main import configure_logging


def main() -> None:
    configure_logging()
    try:
        app = ProbeStationApp(enable_focus_assist=False, enable_autofocus=False)
        app.mainloop()
    except Exception:
        logging.exception("fatal stage 1 GUI error")
        raise


if __name__ == "__main__":
    main()
