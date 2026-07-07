"""
Scale OOT validation from 3 to 300 stocks — direct DBSCAN BUY signal evaluation

Reads all level2_ops CSVs (DBSCAN output), attaches forward returns using
akshare daily OHLC, ranks stocks by BUY signal Alpha quality.

用法:
  # Test with 3 pilot stocks (verify against v6 baseline)
  python scripts/scale_oot_300.py --stocks 002516,301529,300100

  # Full 300-stock run
  python scripts/scale_oot_300.py

  # Resume price download (skip stocks already cached)
  python scripts/scale_oot_300.py --skip-existing-prices
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

OPS_COLS = [
    "cluster_id", "direction", "total_amount_wan", "avg_price",
    "order_count", "time_span_min", "start_time", "end_time",
    "buy_volume_wan", "price_min", "price_max", "vwap_deviation_pct",
    "avg_order_size_wan", "median_order_qty", "qty_cv",
    "mid_time_sec", "order_interval_std", "order_hhi",
    "date", "stock_code", "matched_orders", "match_key",
]


def load_all_ops(ops_dir: Path, stocks_filter: set[str] | None = None) -> pd.DataFrame:
    """加载所有 level2_ops CSV，连接为单个 DataFrame，只保留 BUY"""
    files = sorted(ops_dir.glob("level2_ops_*.csv"))
    if not files:
        print(f"[ERROR] No level2_ops_*.csv found in {ops_dir}")
        sys.exit(1)

    print(f"Loading {len(files)} ops files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  [WARN] skip {f.name}: {e}")

    if not dfs:
        print("[ERROR] No data loaded")
        sys.exit(1)

    ops = pd.concat(dfs, ignore_index=True)
    print(f"  Total ops: {len(ops):,}")

    # Filter BUY only
    ops = ops[ops["direction"] == "BUY"].copy()
    print(f"  BUY ops: {len(ops):,}")

    # Clean stock_code: strip .SZ/.SH suffix, preserve leading zeros
    ops["stock_code_clean"] = ops["stock_code"].str.replace(
        r"\.(SZ|SH)$", "", regex=True
    ).str.zfill(6)

    if stocks_filter:
        ops = ops[ops["stock_code_clean"].isin(stocks_filter)]
        print(f"  After stock filter: {len(ops):,}")

    # Parse date
    ops["date"] = pd.to_datetime(ops["date"].astype(str), format="%Y%m%d")
    ops = ops.sort_values(["stock_code_clean", "date"]).reset_index(drop=True)

    return ops


def prefetch_prices(stocks: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """为所有股票预取日线 OHLC，返回 {stock: DataFrame}"""
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
    """为每条 BUY op 附加基于日线 OHLC 的前向收益

    使用 close-to-close 收益: fwd_Nd = (close_{T+N} / close_T - 1) * 100
    close_T 为信号日收盘价（hfq），与 entry_price (L2实际成交价) 的区别是
    hfq已叠加所有历史分红除权调整，两者不在同一价格坐标系。
    close-to-close 消除此偏差，且对5/10/20日短窗口分红影响极小。
    """
    for h in FWD_HORIZONS:
        ops[f"fwd_{h}d"] = np.nan
        ops[f"win_{h}d"] = np.nan
    # Also store the signal day's hfq close for reference
    ops["close_T"] = np.nan

    for stock, pdf in prices.items():
        mask = ops["stock_code_clean"] == stock
        if not mask.any():
            continue

        stock_ops = ops[mask]
        price_dates = pdf["date"].dt.strftime("%Y%m%d").values
        price_close = pdf["close"].values
        date_to_idx = {d: i for i, d in enumerate(price_dates)}

        for idx in stock_ops.index:
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

    return ops


def evaluate_per_stock(ops: pd.DataFrame) -> pd.DataFrame:
    """按股票聚合 BUY 信号 Alpha 指标"""
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
            vals = buys[fwd_col].dropna()
            wins = buys[win_col].dropna()
            row[f"avg_fwd_{h}d"] = round(vals.mean(), 2) if len(vals) > 0 else None
            row[f"win_{h}d"] = round(wins.mean(), 3) if len(wins) > 0 else None
            row[f"median_fwd_{h}d"] = round(vals.median(), 2) if len(vals) > 0 else None
            row[f"n_fwd_{h}d"] = len(vals)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Composite score: normalize components then weighted sum
    for col in ["win_5d", "avg_fwd_5d", "n_buy_ops"]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].fillna(0)
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                df[f"{col}_norm"] = (vals - vmin) / (vmax - vmin)
            else:
                df[f"{col}_norm"] = 0.0

    norm_cols = [c for c in ["win_5d_norm", "avg_fwd_5d_norm", "n_buy_ops_norm"] if c in df.columns]
    if norm_cols:
        df["score"] = (
            df["win_5d_norm"].fillna(0) * 0.4 +
            df["avg_fwd_5d_norm"].fillna(0) * 0.3 +
            df["n_buy_ops_norm"].fillna(0) * 0.3
        )
    else:
        df["score"] = 0.0

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Scale OOT validation to 300 stocks")
    ap.add_argument("--stocks", default="",
                    help="Comma-separated stock codes to evaluate (default: all in selected_stocks.csv)")
    ap.add_argument("--start-date", default="2025-01-01", help="Start date for price data")
    ap.add_argument("--end-date", default="2025-12-31", help="End date for price data")
    ap.add_argument("--ops-dir", default=str(OPS_DIR), help="Level2 ops directory")
    ap.add_argument("--output-dir", default=str(OOT_DIR), help="Output directory")
    ap.add_argument("--no-prices", action="store_true", help="Skip price fetch (use if already cached)")
    args = ap.parse_args()

    ops_dir = Path(args.ops_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine stock list
    if args.stocks:
        stocks_filter = set(s.strip() for s in args.stocks.split(","))
    else:
        if STOCKS_PATH.exists():
            sel = pd.read_csv(STOCKS_PATH)
            stocks_filter = set(sel["stock"].astype(str).str.zfill(6))
        else:
            print(f"[WARN] {STOCKS_PATH} not found, evaluating all stocks in ops")
            stocks_filter = None

    # Step 1: Load ops
    print("=" * 60)
    print("Step 1: Load level2_ops")
    ops = load_all_ops(ops_dir, stocks_filter)

    # Get unique stocks from ops
    stocks = sorted(ops["stock_code_clean"].unique())
    print(f"  Stocks in ops: {len(stocks)}")

    # Show per-stock signal count distribution
    ops_per_stock = ops.groupby("stock_code_clean").size()
    print(f"  Signals/stock: min={ops_per_stock.min()}, median={ops_per_stock.median():.0f}, "
          f"mean={ops_per_stock.mean():.0f}, max={ops_per_stock.max()}")

    # Step 2: Prefetch prices
    print(f"\n{'='*60}")
    print(f"Step 2: Prefetch daily OHLC for {len(stocks)} stocks")
    if args.no_prices:
        print("  --no-prices set, skipping")
        prices = {}
        # Load from cache
        for stock in stocks:
            cache_path = DAILY_DIR / f"{stock}.parquet"
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                df["date"] = pd.to_datetime(df["date"])
                prices[stock] = df
        print(f"  Loaded {len(prices)} cached price files")
    else:
        prices = prefetch_prices(stocks, args.start_date, args.end_date)

    if not prices:
        print("[ERROR] No price data available. Check network or akshare installation.")
        sys.exit(1)

    # Step 3: Attach forward returns
    print(f"\n{'='*60}")
    print("Step 3: Attach forward returns")
    ops = attach_forward_returns(ops, prices)

    # Per-horizon coverage
    for h in FWD_HORIZONS:
        n_valid = ops[f"fwd_{h}d"].notna().sum()
        mean_fwd = ops[f"fwd_{h}d"].dropna().mean()
        mean_win = ops[f"win_{h}d"].dropna().mean()
        print(f"  fwd_{h}d: {n_valid}/{len(ops)} valid, "
              f"mean_ret={mean_fwd:.2f}%, win_rate={mean_win:.3f}")

    # Step 4: Per-stock evaluation
    print(f"\n{'='*60}")
    print("Step 4: Per-stock evaluation")
    summary = evaluate_per_stock(ops)

    # Print top/bottom
    print(f"\n  Top 10 by score:")
    top_cols = ["stock_code", "n_buy_ops", "avg_fwd_5d", "win_5d", "score"]
    available_top = [c for c in top_cols if c in summary.columns]
    print(summary[available_top].head(10).to_string(index=False))

    # Step 5: Output
    print(f"\n{'='*60}")
    print("Step 5: Output")

    summary_path = out_dir / "oot_300_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path} ({len(summary)} stocks)")

    ops_out = out_dir / "oot_300_ops_with_fwd.csv"
    ops.to_csv(ops_out, index=False)
    print(f"  Ops+forward returns: {ops_out} ({len(ops)} rows, "
          f"{ops_out.stat().st_size / 1_048_576:.1f} MB)")

    # Top 20 deep-dive candidates
    top20 = summary.head(20)["stock_code"].tolist()
    print(f"\n  Top 20 deep-dive candidates: {','.join(top20)}")

    # Distribution stats
    print(f"\n  Score distribution:")
    print(f"    mean={summary['score'].mean():.3f}, median={summary['score'].median():.3f}")
    print(f"    min={summary['score'].min():.3f}, max={summary['score'].max():.3f}")

    for h in FWD_HORIZONS:
        col = f"win_{h}d"
        if col in summary.columns:
            vals = summary[col].dropna()
            if len(vals) > 0:
                print(f"    {col}: mean={vals.mean():.3f}, "
                      f"p25={vals.quantile(0.25):.3f}, p75={vals.quantile(0.75):.3f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
