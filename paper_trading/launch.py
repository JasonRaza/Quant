"""Console entry point for the dashboard (quant-dashboard)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
