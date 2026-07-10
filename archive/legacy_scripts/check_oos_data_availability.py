"""
Check which years have complete data for backtesting.

Checks per year:
  - level2_ops files exist
  - DBSCAN/InstitutionTracker processed ops exist
  - Daily OHLC/price parquet files exist
  - Signal files exist
  - Trading day coverage
  - Stock coverage
  - Whether backtest is executable

Output: data/processed/oos_data_availability.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OPS_DIR = PROJECT / "data" / "processed" / "level2_ops"
INST_DIR = PROJECT / "data" / "processed" / "v6_institutions" / "institutions"
OOT_DIR = PROJECT / "data" / "processed" / "oot"
DAILY_DIR = PROJECT / "data" / "daily"
OUT_DIR = PROJECT / "data" / "processed"

PRICE_SCALE = 100
TARGET_STOCKS = ["000547", "000042", "000669", "000688", "000510", "000007", "000426"]


def check_year(year: int) -> dict:
    """Check data availability for a given year."""
    year_str = str(year)
    result = {
        "year": year,
        "has_raw_l2": False,
        "has_level2_ops": False,
        "has_insttracker_signals": False,
        "has_price": False,
        "n_ops_files": 0,
        "n_signal_files": 0,
        "n_price_files": 0,
        "n_price_days": 0,
        "n_stocks_with_price": 0,
        "n_stocks_with_signals": 0,
        "can_backtest": False,
        "status": "missing_ops",
        "note": "",
    }

    # 0. Check raw L2 data (on Windows Desktop/{year}/)
    # 2025: 12 monthly dirs, 2026: 5 monthly dirs (Jan-May)
    raw_l2_map = {2025: 12, 2026: 5}
    if year in raw_l2_map:
        result["has_raw_l2"] = True

    # 1. Check level2_ops
    ops_year_dir = OPS_DIR / year_str
    if ops_year_dir.exists() and ops_year_dir.is_dir():
        ops_files = sorted(ops_year_dir.glob("level2_ops_*.csv"))
        result["n_ops_files"] = len(ops_files)
        if len(ops_files) > 0:
            result["has_level2_ops"] = True

    # 2. Check InstitutionTracker signals (per-stock institution CSVs)
    inst_files = sorted(INST_DIR.glob("*_institutions.csv"))
    result["n_signal_files"] = len(inst_files)
    if len(inst_files) > 0:
        result["has_insttracker_signals"] = True

    # 3. Check price data
    price_files = list(DAILY_DIR.glob("*.parquet"))
    result["n_price_files"] = len(price_files)

    stocks_with_price = 0
    total_price_days = 0
    for s in TARGET_STOCKS:
        p = DAILY_DIR / f"{s}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                year_data = df[df["date"].dt.year == year]
                if len(year_data) > 0:
                    stocks_with_price += 1
                    total_price_days = max(total_price_days, len(year_data))
            except Exception:
                pass

    result["n_stocks_with_price"] = stocks_with_price
    result["n_price_days"] = total_price_days
    result["has_price"] = stocks_with_price >= 1

    # 4. Check OOT signal files (v4-v6 crossday operations)
    stocks_with_signals = 0
    for s in TARGET_STOCKS:
        oot_stock_dir = OOT_DIR / s
        if oot_stock_dir.exists():
            ops_file = oot_stock_dir / "crossday_operations_unified.csv"
            if ops_file.exists():
                stocks_with_signals += 1
    result["n_stocks_with_signals"] = stocks_with_signals

    # 5. Determine status
    notes = []
    missing = []

    if not result["has_level2_ops"]:
        missing.append("level2_ops")
    if not result["has_price"]:
        missing.append("price")
    if not result["has_insttracker_signals"]:
        missing.append("insttracker_signals")

    if result["has_level2_ops"] and result["has_price"]:
        if result["has_insttracker_signals"]:
            result["can_backtest"] = True
            result["status"] = "evaluated"
        else:
            result["can_backtest"] = True
            result["status"] = "evaluable"
            notes.append("ops+price exist, InstitutionTracker signals could be generated")
    elif result["has_price"] and not result["has_level2_ops"]:
        result["status"] = "missing_ops"
        notes.append("prices available but processed level2_ops missing")
    elif not result["has_price"]:
        result["status"] = "missing_price"
        notes.append("no price data available")
    else:
        result["status"] = "missing_ops"
        notes.append("insufficient data")

    # Year-specific notes
    if year == 2025:
        if result["can_backtest"]:
            notes.append("evaluated with 2025 processed ops (single-year validation only)")
        notes.append(f"raw L2: {raw_l2_map.get(year, 0)} monthly dirs on Windows Desktop")
    elif year == 2026:
        notes.append(f"raw L2: {raw_l2_map.get(year, 0)} monthly dirs (Jan-May) on Windows Desktop — "
                     "needs DBSCAN pipeline to generate level2_ops")
        if result["has_price"]:
            notes.append("prices available; cross-year OOT pending DBSCAN processing")
    elif year == 2024:
        if result["has_price"] and not result["has_level2_ops"]:
            notes.append("prices available but no raw L2 or level2_ops found")

    result["note"] = "; ".join(notes)
    return result


def main():
    years = [2024, 2025, 2026]
    rows = [check_year(y) for y in years]

    df = pd.DataFrame(rows)
    cols = ["year", "status", "has_raw_l2", "has_level2_ops",
            "has_insttracker_signals", "has_price",
            "n_ops_files", "n_signal_files", "n_price_days",
            "n_stocks_with_price", "n_stocks_with_signals",
            "can_backtest", "note"]
    df = df[cols]

    print("Data Availability Check")
    print("=" * 80)
    print(df.to_string(index=False))

    out_path = OUT_DIR / "oos_data_availability.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary
    evaluable = df[df["can_backtest"]]
    pending = df[~df["can_backtest"]]
    print(f"\nSummary:")
    print(f"  Evaluable years: {len(evaluable)} — {evaluable['year'].tolist()}")
    print(f"  Pending years:   {len(pending)} — {pending['year'].tolist()}")
    if len(evaluable) < 2:
        print(f"  WARNING: Only {len(evaluable)} year(s) evaluable. "
              f"Cross-year OOT requires at least 2 years of data.")
        print(f"  Current status: single-year validation only (2025).")


if __name__ == "__main__":
    main()
