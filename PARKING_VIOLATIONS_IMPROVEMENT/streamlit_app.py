"""Streamlit Community Cloud entrypoint.

Keep this as a stable main script for Streamlit deployment.
Delegates to the real app in app.py without relying on import paths.
"""

from __future__ import annotations

from pathlib import Path
import runpy

_APP = Path(__file__).resolve().parent / "app.py"
runpy.run_path(_APP.as_posix(), run_name="__main__")
