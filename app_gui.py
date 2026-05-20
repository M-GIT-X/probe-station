"""Compatibility shim for the unified GUI module.

The project now uses gui_app.py as the single GUI implementation. This module is
kept only so older imports do not break.
"""

from gui_app import *  # noqa: F401,F403
