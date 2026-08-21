"""Compatibility entry point for the automatic EIS module.

The release launcher prepares one shared application environment.  This file
therefore delegates directly to ``eis_auto.py`` instead of creating a second
virtual environment and reinstalling the same scientific dependencies.
"""

from pathlib import Path
import os
import runpy
import sys


APP_DIR = Path(__file__).resolve().parent
TARGET = APP_DIR / "eis_auto.py"


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Automatic EIS module not found: {TARGET}")

    os.chdir(APP_DIR)
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
