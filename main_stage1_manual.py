"""Legacy entry point.

The stage-specific startup flow is retired. Run main.py for the unified GUI.
"""

from __future__ import annotations

import main as unified_main


def main() -> None:
    unified_main.main()


if __name__ == "__main__":
    main()
