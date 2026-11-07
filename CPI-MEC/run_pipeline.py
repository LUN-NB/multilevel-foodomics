"""Launch the formal CPI-MEC Adaptive pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Load and run the formal workflow."""
    root = Path(__file__).resolve().parent
    core_dir = root / "src" / "cpi_mec"
    sys.path.insert(0, str(core_dir))

    from adaptive_workflow import main as run_workflow

    run_workflow()


if __name__ == "__main__":
    main()
