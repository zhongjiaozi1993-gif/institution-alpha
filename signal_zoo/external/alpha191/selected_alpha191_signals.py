"""Selected GTJA191 Alpha factors for Sprint 2.

Only 4 factors are activated. Do NOT expose all 191.
Use adapter.py for the underlying computation.
"""
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent.parent
SIGNAL_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

from .adapter import compute_signal_batch, FACTOR_REGISTRY

# Sprint 2 selected factors: 4 GTJA191 formulas
# Ordered by category diversity: reversal, volume-price, volatility, momentum
SELECTED = [
    "gtja_002",   # Signal017: Reversal (Williams %R)
    "gtja_004",   # Signal018: Volume-Price (trend regime)
    "gtja_070",   # Signal019: Volatility (amount std)
    "gtja_085",   # Signal020: Momentum (ts_rank × price delta)
]


def generate_all_signals(
    stock_codes: list[str],
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    zscore: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate signal DataFrames for all 4 selected factors.

    Returns dict: factor_key -> DataFrame
    """
    results = {}
    for fk in SELECTED:
        info = FACTOR_REGISTRY[fk]
        print(f"  Computing {info['signal_id']} ({info['signal_name']})...")
        df = compute_signal_batch(stock_codes, fk, start_date, end_date, zscore=zscore)
        # Save parquet
        signal_id = info["signal_id"].lower()
        out_path = SIGNAL_DIR / f"{signal_id}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"    -> {out_path} ({len(df)} rows)")
        results[fk] = df
    return results


def load_candidate_stocks() -> list[str]:
    """Load V0 candidate stocks from the universe file."""
    candidate_file = PROJECT / "data" / "processed" / "stock_universe" / "dbscan_candidate_v0.txt"
    stocks = []
    with open(candidate_file) as f:
        for line in f:
            line = line.strip()
            if line:
                stocks.append(line.zfill(6))
    return stocks


def main():
    stocks = load_candidate_stocks()
    print(f"Generating signals for {len(stocks)} candidate stocks...")
    generate_all_signals(stocks)
    print("Done.")


if __name__ == "__main__":
    main()
