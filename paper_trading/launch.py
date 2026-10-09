"""Console entry point for the dashboard (quant-dashboard)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    dashboard = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
