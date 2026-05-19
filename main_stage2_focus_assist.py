"""Stage 2 entry point for manual focus assist."""

from __future__ import annotations

import logging

from app_gui import ProbeStationApp
from main import configure_logging


def main() -> None:
    configure_logging()
    try:
        app = ProbeStationApp(enable_focus_assist=True, enable_autofocus=False)
        app.mainloop()
    except Exception:
        logging.exception("fatal stage 2 GUI error")
        raise


if __name__ == "__main__":
    main()
