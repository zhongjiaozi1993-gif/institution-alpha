"""Selected GTJA191 Alpha factors for Sprint 3.

30 factors (4 Sprint 2 + 26 Sprint 3) across 5 categories.
Use adapter.py for the underlying computation.
"""
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent.parent
SIGNAL_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191_full"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

from .adapter import compute_signal_batch, FACTOR_REGISTRY

# Sprint 2 (4) + Sprint 3 (26) = 30 factors
SELECTED = [
    # Sprint 2 (Signal017-020)
    "gtja_002",   # Signal017: Reversal (Williams %R)
    "gtja_004",   # Signal018: Volume-Price (trend regime)
    "gtja_070",   # Signal019: Volatility (amount std)
    "gtja_085",   # Signal020: Momentum (ts_rank × price delta)
    # Sprint 3: Momentum (Signal021-026)
    "gtja_014",   # 5d price momentum
    "gtja_053",   # 12d up-day pct
    "gtja_088",   # 20d pct change
    "gtja_106",   # 20d absolute change
    "gtja_112",   # Chande Momentum Oscillator
    "gtja_167",   # 12d cumulative up-move
    # Sprint 3: Reversal (Signal027-030)
    "gtja_046",   # Multi-MA price ratio
    "gtja_065",   # MA6 / price
    "gtja_066",   # Pct deviation from MA6
    "gtja_078",   # CCI-type oscillator
    # Sprint 3: Volume-Price (Signal031-037)
    "gtja_011",   # Volume-weighted intraday position
    "gtja_032",   # Ranked hi-vol correlation
    "gtja_084",   # Signed volume (buying pressure)
    "gtja_102",   # Volume RSI
    "gtja_128",   # Money Flow Index
    "gtja_150",   # TP × log(volume)
    "gtja_178",   # Volume-weighted return
    # Sprint 3: Volatility (Signal038-042)
    "gtja_049",   # Asymmetric range
    "gtja_076",   # CV of vol-adj returns
    "gtja_095",   # 20d amount std
    "gtja_158",   # Normalized daily range
    "gtja_161",   # Avg True Range
    # Sprint 3: Trend/MA (Signal043-046)
    "gtja_089",   # MACD-type
    "gtja_096",   # Double-smoothed Stochastic
    "gtja_153",   # BBI (4-MA average)
    "gtja_172",   # ADX-type
]


def generate_all_signals(
    stock_codes: list[str],
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    zscore: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate signal DataFrames for all 30 selected factors."""
    results = {}
    for fk in SELECTED:
        info = FACTOR_REGISTRY[fk]
        print(f"  Computing {info['signal_id']} ({info['signal_name']})...")
        df = compute_signal_batch(stock_codes, fk, start_date, end_date, zscore=zscore)
        signal_id = info["signal_id"].lower()
        out_path = SIGNAL_DIR / f"{signal_id}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"    -> {out_path} ({len(df)} rows)")
        results[fk] = df
    return results


def load_validation_300_stocks() -> list[str]:
    """Load 300-stock validation universe (only stocks with daily data). Legacy."""
    val_file = PROJECT / "data" / "processed" / "stock_universe" / "validation_300.txt"
    stocks = []
    with open(val_file) as f:
        for line in f:
            code, status = line.strip().split()
            if status == "ok":
                stocks.append(code.zfill(6))
    return stocks


def load_alpha191_daily_universe() -> list[str]:
    """Load Alpha191 daily universe (Universe002): ZZ1000 + ZZ500 with daily data."""
    uni_file = PROJECT / "data" / "processed" / "stock_universe" / "daily_alpha191_full.txt"
    stocks = []
    with open(uni_file) as f:
        for line in f:
            code, status = line.strip().split()
            if status == "ok":
                stocks.append(code.zfill(6))
    return stocks


def load_candidate_stocks() -> list[str]:
    """Load V0 candidate stocks (legacy, 13 stocks)."""
    candidate_file = PROJECT / "data" / "processed" / "stock_universe" / "dbscan_candidate_v0.txt"
    stocks = []
    with open(candidate_file) as f:
        for line in f:
            line = line.strip()
            if line:
                stocks.append(line.zfill(6))
    return stocks


def main():
    stocks = load_alpha191_daily_universe()
    print(f"Generating signals for {len(stocks)} Alpha191 daily universe stocks (Universe002)...")
    generate_all_signals(stocks)
    print("Done.")


if __name__ == "__main__":
    main()
