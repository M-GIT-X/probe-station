"""Stage 3 entry point for conservative Z-only autofocus."""

from __future__ import annotations

import logging

from app_gui import ProbeStationApp
from main import configure_logging


def main() -> None:
    configure_logging()
    try:
        app = ProbeStationApp(enable_focus_assist=True, enable_autofocus=True)
        app.mainloop()
    except Exception:
        logging.exception("fatal stage 3 GUI error")
        raise


if __name__ == "__main__":
    main()
