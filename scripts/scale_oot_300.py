"""
Scale OOT validation — DBSCAN BUY signal evaluation with excess returns.

用法:
  # Priority 25 stocks
  python scripts/scale_oot_300.py --stock-file config/v6_priority_stocks.txt

  # Specific stocks
  python scripts/scale_oot_300.py --stocks 002516,301529,300100

  # Full 300-stock run
  python scripts/scale_oot_300.py

  # Skip price download (use cache)
  python scripts/scale_oot_300.py --no-prices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.data.price_loader import load_stock_daily

FWD_HORIZONS = [1, 3, 5, 10, 20]
OPS_DIR = PROJECT / "data" / "processed" / "level2_ops" / "2025"
OOT_DIR = PROJECT / "data" / "processed" / "oot"
STOCKS_PATH = PROJECT / "data" / "processed" / "stock_universe" / "selected_stocks.csv"
DAILY_DIR = PROJECT / "data" / "daily"


def load_stocks_from_file(path: str) -> set[str]:
    """Read stock codes from file (one per line, or CSV)."""
    p = Path(path)
    if not p.exists():
        p = PROJECT / path
    if not p.exists():
        print(f"[ERROR] Stock file not found: {path}")
        sys.exit(1)

    codes = set()
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        for col in ["stock_code", "stock", "code"]:
            if col in df.columns:
                codes.update(df[col].astype(str).str.replace(r"\.(SZ|SH)$", "", regex=True).str.zfill(6))
                break
        else:
            codes.update(df.iloc[:, 0].astype(str).str.replace(r"\.(SZ|SH)$", "", regex=True).str.zfill(6))
    else:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    codes.add(line.replace(".SZ", "").replace(".SH", "").zfill(6))
    return codes


def load_all_ops(ops_dir: Path, stocks_filter: set[str] | None = None) -> pd.DataFrame:
    """Load all level2_ops CSVs, concat, keep BUY only."""
    files = sorted(ops_dir.glob("level2_ops_*.csv"))
    if not files:
        print(f"[ERROR] No level2_ops_*.csv found in {ops_dir}")
        sys.exit(1)

    print(f"Loading {len(files)} ops files...")
    dfs = []
    bad_files = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            bad_files.append((f.name, str(e)[:80]))

    if bad_files:
        print(f"  Skipped {len(bad_files)} bad files:")
        for name, err in bad_files[:5]:
            print(f"    {name}: {err}")

    if not dfs:
        print("[ERROR] No data loaded")
        sys.exit(1)

    ops = pd.concat(dfs, ignore_index=True)
    print(f"  Total ops: {len(ops):,}")

    ops = ops[ops["direction"] == "BUY"].copy()
    print(f"  BUY ops: {len(ops):,}")

    ops["stock_code_clean"] = ops["stock_code"].str.replace(
        r"\.(SZ|SH)$", "", regex=True
    ).str.zfill(6)

    if stocks_filter:
        ops = ops[ops["stock_code_clean"].isin(stocks_filter)]
        print(f"  After stock filter: {len(ops):,}")

    ops["date"] = pd.to_datetime(ops["date"].astype(str), format="%Y%m%d")
    ops = ops.sort_values(["stock_code_clean", "date"]).reset_index(drop=True)
    return ops


def prefetch_prices(stocks: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLC for all stocks, return {stock: DataFrame}."""
    prices = {}
    n = len(stocks)
    t0 = time.time()

    for i, stock in enumerate(stocks):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  Price fetch: {i+1}/{n} ({elapsed:.0f}s)")

        df = load_stock_daily(stock, start_date=start_date, end_date=end_date, adjust="hfq")
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            prices[stock] = df

    print(f"  Price fetch done: {len(prices)}/{n} stocks ({time.time() - t0:.0f}s)")
    return prices


def attach_forward_returns(ops: pd.DataFrame,
                           prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach forward returns and cross-sectional excess returns.

    fwd_Nd = (close_{T+N} / close_T - 1) * 100   (close-to-close, hfq)
    bench_fwd_Nd = same_date universe average fwd_Nd
    excess_fwd_Nd = fwd_Nd - bench_fwd_Nd
    """
    for h in FWD_HORIZONS:
        ops[f"fwd_{h}d"] = np.nan
        ops[f"win_{h}d"] = np.nan
        ops[f"bench_fwd_{h}d"] = np.nan
        ops[f"excess_fwd_{h}d"] = np.nan
    ops["close_T"] = np.nan

    # Step 1: compute stock-level forward returns
    for stock, pdf in prices.items():
        mask = ops["stock_code_clean"] == stock
        if not mask.any():
            continue

        stock_ops_idx = ops[mask].index
        price_dates = pdf["date"].dt.strftime("%Y%m%d").values
        price_close = pdf["close"].values
        date_to_idx = {d: i for i, d in enumerate(price_dates)}

        for idx in stock_ops_idx:
            op_date = ops.loc[idx, "date"].strftime("%Y%m%d")
            if op_date not in date_to_idx:
                continue

            di = date_to_idx[op_date]
            close_t = price_close[di]
            if close_t <= 0:
                continue

            ops.at[idx, "close_T"] = close_t

            for h in FWD_HORIZONS:
                ti = di + h
                if ti < len(price_close):
                    ret = (price_close[ti] / close_t - 1) * 100
                    ops.at[idx, f"fwd_{h}d"] = round(ret, 3)
                    ops.at[idx, f"win_{h}d"] = 1.0 if ret > 0 else 0.0

    # Step 2: compute benchmark = same-date cross-sectional average (equal-weight)
    for h in FWD_HORIZONS:
        bench = ops.groupby("date")[f"fwd_{h}d"].transform("mean")
        ops[f"bench_fwd_{h}d"] = bench
        ops[f"excess_fwd_{h}d"] = ops[f"fwd_{h}d"] - bench

    return ops


def evaluate_per_stock(ops: pd.DataFrame) -> pd.DataFrame:
    """Per-stock aggregation of BUY signal alpha with excess return metrics."""
    rows = []
    for stock, g in ops.groupby("stock_code_clean"):
        buys = g[g["direction"] == "BUY"]
        n = len(buys)

        row = {
            "stock_code": stock,
            "n_buy_ops": n,
            "n_days": buys["date"].nunique(),
            "avg_daily_ops": round(n / max(1, buys["date"].nunique()), 1),
            "avg_amount_wan": round(buys["total_amount_wan"].mean(), 1),
            "total_buy_wan": round(buys["total_amount_wan"].sum(), 1),
            "avg_order_count": round(buys["order_count"].mean(), 1),
            "avg_time_span_min": round(buys["time_span_min"].mean(), 1),
            "avg_vwap_deviation_pct": round(buys["vwap_deviation_pct"].mean(), 2),
        }

        for h in FWD_HORIZONS:
            fwd_col = f"fwd_{h}d"
            win_col = f"win_{h}d"
            bench_col = f"bench_fwd_{h}d"
            excess_col = f"excess_fwd_{h}d"

            vals = buys[fwd_col].dropna()
            excess_vals = buys[excess_col].dropna()
            wins = buys[win_col].dropna()

            row[f"avg_fwd_{h}d"] = round(vals.mean(), 2) if len(vals) > 0 else None
            row[f"win_{h}d"] = round(wins.mean(), 3) if len(wins) > 0 else None
            row[f"median_fwd_{h}d"] = round(vals.median(), 2) if len(vals) > 0 else None
            row[f"n_fwd_{h}d"] = len(vals)

            # Excess return metrics
            row[f"avg_excess_fwd_{h}d"] = round(excess_vals.mean(), 2) if len(excess_vals) > 0 else None
            row[f"std_excess_fwd_{h}d"] = round(excess_vals.std(), 2) if len(excess_vals) > 1 else None
            # excess_win: positive excess return = win
            if len(excess_vals) > 0:
                row[f"excess_win_{h}d"] = round((excess_vals > 0).mean(), 3)
            else:
                row[f"excess_win_{h}d"] = None

        # Stability: avg_excess_fwd_5d / std_excess_fwd_5d (like Sharpe)
        avg_ex5 = row.get("avg_excess_fwd_5d")
        std_ex5 = row.get("std_excess_fwd_5d")
        if avg_ex5 is not None and std_ex5 is not None and std_ex5 > 0:
            row["stability"] = round(avg_ex5 / max(std_ex5, 0.01), 3)
        else:
            row["stability"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Composite score: excess-return focused ----
    score_cols = {
        "avg_excess_fwd_5d": 0.4,
        "excess_win_5d": 0.3,
        "n_buy_ops": 0.2,
        "stability": 0.1,
    }

    for col, weight in score_cols.items():
        if col in df.columns and df[col].notna().any():
            vals = df[col].fillna(0)
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                df[f"{col}_norm"] = (vals - vmin) / (vmax - vmin)
            else:
                df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = 0.0

    df["score"] = (
        df["avg_excess_fwd_5d_norm"].fillna(0) * 0.4 +
        df["excess_win_5d_norm"].fillna(0) * 0.3 +
        df["n_buy_ops_norm"].fillna(0) * 0.2 +
        df["stability_norm"].fillna(0) * 0.1
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Scale OOT validation with excess returns")
    ap.add_argument("--stocks", default="",
                    help="Comma-separated stock codes")
    ap.add_argument("--stock-file", default="",
                    help="Path to file with stock codes (txt or csv)")
    ap.add_argument("--start-date", default="2025-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--ops-dir", default=str(OPS_DIR))
    ap.add_argument("--output-dir", default=str(OOT_DIR))
    ap.add_argument("--no-prices", action="store_true")
    ap.add_argument("--output-prefix", default="oot_300",
                    help="Output file prefix (e.g. oot_priority25)")
    args = ap.parse_args()

    ops_dir = Path(args.ops_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine stock list
    stocks_filter = None
    if args.stocks:
        stocks_filter = set(s.strip() for s in args.stocks.split(","))
    elif args.stock_file:
        stocks_filter = load_stocks_from_file(args.stock_file)

    if stocks_filter is None and STOCKS_PATH.exists():
        sel = pd.read_csv(STOCKS_PATH)
        stocks_filter = set(sel["stock"].astype(str).str.zfill(6))

    # Step 1: Load ops
    print("=" * 60)
    print("Step 1: Load level2_ops")
    ops = load_all_ops(ops_dir, stocks_filter)

    stocks = sorted(ops["stock_code_clean"].unique())
    print(f"  Stocks in ops: {len(stocks)}")

    ops_per_stock = ops.groupby("stock_code_clean").size()
    print(f"  Signals/stock: min={ops_per_stock.min()}, median={ops_per_stock.median():.0f}, "
          f"mean={ops_per_stock.mean():.0f}, max={ops_per_stock.max()}")

    # Step 2: Prices
    print(f"\n{'='*60}")
    print(f"Step 2: Daily OHLC for {len(stocks)} stocks")
    if args.no_prices:
        print("  --no-prices, loading from cache")
        prices = {}
        missing = []
        for stock in stocks:
            cache_path = DAILY_DIR / f"{stock}.parquet"
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                df["date"] = pd.to_datetime(df["date"])
                prices[stock] = df
            else:
                missing.append(stock)
        print(f"  Loaded {len(prices)} cached, {len(missing)} missing")
        if missing:
            print(f"  Missing: {', '.join(missing[:10])}")
    else:
        prices = prefetch_prices(stocks, args.start_date, args.end_date)

    if not prices:
        print("[ERROR] No price data. Check network or akshare.")
        sys.exit(1)

    # Step 3: Forward + excess returns
    print(f"\n{'='*60}")
    print("Step 3: Forward + excess returns")
    ops = attach_forward_returns(ops, prices)

    for h in FWD_HORIZONS:
        n_valid = ops[f"fwd_{h}d"].notna().sum()
        n_excess = ops[f"excess_fwd_{h}d"].notna().sum()
        mean_fwd = ops[f"fwd_{h}d"].dropna().mean()
        mean_win = ops[f"win_{h}d"].dropna().mean()
        mean_excess = ops[f"excess_fwd_{h}d"].dropna().mean()
        print(f"  fwd_{h}d: {n_valid}/{len(ops)} valid, "
              f"mean_ret={mean_fwd:+.2f}%, win_rate={mean_win:.3f}, "
              f"mean_excess={mean_excess:+.2f}%")

    # Step 4: Per-stock evaluation
    print(f"\n{'='*60}")
    print("Step 4: Per-stock evaluation")
    summary = evaluate_per_stock(ops)

    top_cols = ["stock_code", "n_buy_ops", "avg_fwd_5d", "win_5d",
                "avg_excess_fwd_5d", "excess_win_5d", "stability", "score"]
    available_top = [c for c in top_cols if c in summary.columns]
    print(f"\n  Top 10 by score:")
    print(summary[available_top].head(10).to_string(index=False))

    # Step 5: Output
    print(f"\n{'='*60}")
    print("Step 5: Output")

    prefix = args.output_prefix
    summary_path = out_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path} ({len(summary)} stocks)")

    ops_out = out_dir / f"{prefix}_ops_with_fwd.csv"
    ops.to_csv(ops_out, index=False)
    out_mb = ops_out.stat().st_size / 1_048_576 if ops_out.exists() else 0
    print(f"  Ops+forward: {ops_out} ({len(ops)} rows, {out_mb:.1f} MB)")

    top20 = summary.head(20)["stock_code"].tolist()
    print(f"\n  Top 20: {','.join(str(c) for c in top20)}")

    print(f"\n  Score distribution:")
    print(f"    mean={summary['score'].mean():.3f}, median={summary['score'].median():.3f}")
    print(f"    min={summary['score'].min():.3f}, max={summary['score'].max():.3f}")

    for h in FWD_HORIZONS:
        for metric in ["win", "excess_win"]:
            col = f"{metric}_{h}d"
            if col in summary.columns:
                vals = summary[col].dropna()
                if len(vals) > 0:
                    print(f"    {col}: mean={vals.mean():.3f}, "
                          f"p25={vals.quantile(0.25):.3f}, p75={vals.quantile(0.75):.3f}")

    # Missing data report
    missing_prices = [s for s in stocks if s not in prices]
    if missing_prices:
        print(f"\n  Stocks missing price data: {len(missing_prices)}")
        for mp in missing_prices[:10]:
            n_ops = len(ops[ops["stock_code_clean"] == mp])
            print(f"    {mp}: {n_ops} BUY ops skipped")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
