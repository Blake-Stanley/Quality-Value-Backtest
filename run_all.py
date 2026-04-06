"""
Run the full backtest pipeline: backtest, make table, then holdings snapshot.

Usage (from repo root):
    python run_all.py              # full pipeline
    python Code/export_holdings.py # holdings snapshot only (uses cache if available)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Code"))

import backtest as bt
import export_holdings as eh
import make_table as mt

if __name__ == "__main__":
    t0 = time.time()
    bt.main()
    print()
    mt.main()
    print()
    eh.main()
    print(f"\nAll done in {time.time() - t0:.1f}s")
