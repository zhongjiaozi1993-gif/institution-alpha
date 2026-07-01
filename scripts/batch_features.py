"""
Batch feature computation for all days of 002516.
Processes all extracted CSV directories and computes 34 features per day.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.level2_reader import read_level2_stock_dir, match_orders_to_trades
from src.cluster.split_detector import detect_institution_operations
from scripts.single_stock_pipeline import compute_enriched_features

STOCK = "002516"
STOCK_WIND = "002516.SZ"
PROJECT = Path(__file__).parent.parent
RAW_ROOT = PROJECT / "data" / "single_stock" / STOCK / "raw"
FEAT_DIR = PROJECT / "data" / "single_stock" / STOCK / "features"
OPS_DIR = PROJECT / "data" / "single_stock" / STOCK / "ops"

for d in [FEAT_DIR, OPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def process_day(stock_dir: Path) -> dict | None:
    """Process one day: load → match → cluster → features. Returns features dict."""
    date = stock_dir.parent.name

    try:
        data = read_level2_stock_dir(stock_dir)
    except Exception as e:
        return None

    orders = data.get("逐笔委托", pd.DataFrame())
    trades = data.get("逐笔成交", pd.DataFrame())
    if orders.empty or trades.empty:
        return None

    matched = match_orders_to_trades(orders, trades)
    if matched.empty:
        return None

    # Try eps sweep
    clusters = []
    for eps in [0.05, 0.10, 0.15, 0.25]:
        for min_samp in [3, 5]:
            clusters = detect_institution_operations(
                matched, eps=eps, min_samples=min_samp, min_total_amount_wan=50,
            )
            if clusters:
                break
        if clusters:
            break

    features = compute_enriched_features(orders, trades, matched, clusters, date)

    # Save ops with date tag
    for c in clusters:
        c["date"] = date
        c["stock_code"] = STOCK

    return {"features": features, "clusters": clusters, "date": date, "n_orders": len(orders), "n_trades": len(trades)}


def main():
    stock_dirs = sorted(RAW_ROOT.glob(f"*/{STOCK_WIND}"))
    print(f"Found {len(stock_dirs)} days to process")

    all_features = []
    all_ops = []
    errors = 0
    t0 = time.time()

    for i, stock_dir in enumerate(stock_dirs):
        result = process_day(stock_dir)
        if result is None:
            errors += 1
            continue

        all_features.append(result["features"])
        all_ops.extend(result["clusters"])

        # Save daily features
        feat_df = pd.DataFrame([result["features"]])
        feat_df.to_csv(FEAT_DIR / f"features_{result['date']}.csv", index=False)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(stock_dirs) - i - 1) / rate / 60
            print(f"  {i+1}/{len(stock_dirs)}: {result['date']} "
                  f"| {result['n_orders']} orders, {len(result['clusters'])} clusters "
                  f"| {rate:.1f}/s | ~{remaining:.0f}min left")

    elapsed = time.time() - t0
    print(f"\nDone {len(all_features)} days in {elapsed/60:.1f} min ({errors} errors)")

    # Save summaries
    if all_ops:
        pd.DataFrame(all_ops).to_csv(OPS_DIR / "all_ops.csv", index=False)
        print(f"Total ops: {len(all_ops)}")

    if all_features:
        summary = pd.DataFrame(all_features)
        summary.to_csv(FEAT_DIR / "all_features.csv", index=False)
        print(f"Features: {len(summary)} days × {len(summary.columns)} columns")

        # Basic stats
        print(f"\nFeature distributions (mean ± std):")
        for c in summary.columns:
            if c in ("date", "stock"):
                continue
            vals = summary[c].dropna()
            if len(vals) > 1 and vals.std() > 0:
                print(f"  {c}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    main()
