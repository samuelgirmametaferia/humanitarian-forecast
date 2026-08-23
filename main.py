#!/usr/bin/env python3
"""Project entry point.

Examples:
  python main.py systems
  python main.py workflow plan
  python main.py workflow full --skip-download
  python main.py models info location/candidate_ranker v9
  python main.py run location.predict -- --index -1 --top 5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanitarian_forecast.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
