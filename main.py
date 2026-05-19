"""Application entry point."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from app_gui import ProbeStationApp


def configure_logging() -> None:
    log_path = Path(__file__).with_name("debug.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def main() -> None:
    configure_logging()
    try:
        app = ProbeStationApp()
        app.mainloop()
    except Exception:
        logging.exception("fatal GUI error")
        raise


if __name__ == "__main__":
    main()
