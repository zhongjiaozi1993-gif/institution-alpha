"""
Scale OOT validation — DBSCAN BUY signal evaluation with multi-benchmark excess returns.

用法:
  python scripts/scale_oot_300.py --stock-file config/v6_priority_stocks.txt
  python scripts/scale_oot_300.py --stocks 002516,301529,300100
  python scripts/scale_oot_300.py --stock-file data/processed/stock_universe/zz1000_liquid_top100.txt --output-prefix oot_top100 --no-prices

Benchmark 口径:
  signal_bench  — 同日期所有BUY信号的等权平均收益
  universe_bench — 同日期Universe全部股票的等权平均收益
  index_bench   — 预留，暂用中证1000（待接入指数数据）
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
        print(f"  Skipped {len(bad_files)} bad files")

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
    prices = {}
    n = len(stocks)
    t0 = time.time()
    for i, stock in enumerate(stocks):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Price fetch: {i+1}/{n} ({time.time()-t0:.0f}s)")
        df = load_stock_daily(stock, start_date=start_date, end_date=end_date, adjust="hfq")
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            prices[stock] = df
    print(f"  Price fetch done: {len(prices)}/{n} stocks ({time.time()-t0:.0f}s)")
    return prices


def attach_forward_returns(ops: pd.DataFrame,
                           prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach forward returns, multi-benchmark excess returns, and entry_price returns.

    Returns computed:
      fwd_Nd           — close-to-close forward return (hfq)
      signal_bench_fwd_Nd  — same-date BUY signal equal-weight avg
      signal_excess_fwd_Nd — fwd - signal_bench
      universe_bench_fwd_Nd — same-date universe ALL stocks equal-weight avg
      universe_excess_fwd_Nd — fwd - universe_bench
      entry_fwd_Nd     — close-to-entry_price return (experimental)
    """
    all_stocks = sorted(prices.keys())

    # --------------------------------
    # Step 1: compute full panel of daily forward returns for ALL universe stocks
    # Used for universe_bench
    # --------------------------------
    print("  Building universe forward return panel...")
    panel_rows = []
    for stock in all_stocks:
        pdf = prices[stock]
        closes = pdf["close"].values
        dates = pdf["date"].dt.strftime("%Y%m%d").values
        n = len(closes)
        for i in range(n):
            row = {"stock": stock, "date": dates[i]}
            close_t = closes[i]
            if close_t <= 0:
                continue
            for h in FWD_HORIZONS:
                ti = i + h
                if ti < n:
                    row[f"fwd_{h}d"] = (closes[ti] / close_t - 1) * 100
            if all(f"fwd_{h}d" in row for h in FWD_HORIZONS):
                panel_rows.append(row)

    panel = pd.DataFrame(panel_rows)
    # universe_bench per date = equal-weight avg across all stocks
    universe_bench = panel.groupby("date")[[f"fwd_{h}d" for h in FWD_HORIZONS]].mean()
    universe_bench = universe_bench.rename(columns={f"fwd_{h}d": f"universe_bench_fwd_{h}d" for h in FWD_HORIZONS})
    print(f"  Universe panel: {len(panel):,} rows, {len(all_stocks)} stocks, "
          f"{universe_bench.index.nunique()} dates")

    # --------------------------------
    # Step 2: per-signal forward returns
    # --------------------------------
    for h in FWD_HORIZONS:
        ops[f"fwd_{h}d"] = np.nan
        ops[f"win_{h}d"] = np.nan
        ops[f"signal_bench_fwd_{h}d"] = np.nan
        ops[f"signal_excess_fwd_{h}d"] = np.nan
        ops[f"universe_bench_fwd_{h}d"] = np.nan
        ops[f"universe_excess_fwd_{h}d"] = np.nan
        ops[f"index_bench_fwd_{h}d"] = np.nan    # reserved
        ops[f"index_excess_fwd_{h}d"] = np.nan   # reserved
        ops[f"entry_fwd_{h}d"] = np.nan
        ops[f"entry_win_{h}d"] = np.nan
    ops["close_T"] = np.nan

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
            entry_price = ops.loc[idx, "avg_price"]

            for h in FWD_HORIZONS:
                ti = di + h
                if ti < len(price_close):
                    future_close = price_close[ti]
                    # close-to-close (hfq)
                    ret = (future_close / close_t - 1) * 100
                    ops.at[idx, f"fwd_{h}d"] = round(ret, 3)
                    ops.at[idx, f"win_{h}d"] = 1.0 if ret > 0 else 0.0

            # entry_price returns: REQUIRES qfq close prices
            # avg_price (~11 yuan actual trade) and hfq close (~1500 yuan for 000001)
            # are in incompatible price coordinate systems.
            # entry_fwd fields remain NaN until qfq prices are available.
            # See oot_top100_report.md for details.

    # --------------------------------
    # Step 3: signal_bench (same-date BUY signal avg)
    # --------------------------------
    for h in FWD_HORIZONS:
        bench = ops.groupby("date")[f"fwd_{h}d"].transform("mean")
        ops[f"signal_bench_fwd_{h}d"] = bench
        ops[f"signal_excess_fwd_{h}d"] = ops[f"fwd_{h}d"] - bench

    # --------------------------------
    # Step 4: merge universe_bench
    # --------------------------------
    ops_date_str = ops["date"].dt.strftime("%Y%m%d")
    for h in FWD_HORIZONS:
        col = f"universe_bench_fwd_{h}d"
        bench_map = universe_bench[col]
        ops[col] = ops_date_str.map(bench_map)
        ops[f"universe_excess_fwd_{h}d"] = ops[f"fwd_{h}d"] - ops[col]

    return ops


def evaluate_per_stock(ops: pd.DataFrame) -> pd.DataFrame:
    """Per-stock aggregation with all benchmark and entry_price metrics."""
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
            signal_ex_col = f"signal_excess_fwd_{h}d"
            universe_ex_col = f"universe_excess_fwd_{h}d"
            entry_col = f"entry_fwd_{h}d"
            entry_win_col = f"entry_win_{h}d"

            vals = buys[fwd_col].dropna()
            row[f"avg_fwd_{h}d"] = round(vals.mean(), 2) if len(vals) > 0 else None
            row[f"win_{h}d"] = round(buys[win_col].dropna().mean(), 3) if len(vals) > 0 else None
            row[f"median_fwd_{h}d"] = round(vals.median(), 2) if len(vals) > 0 else None
            row[f"n_fwd_{h}d"] = len(vals)

            # signal excess
            se_vals = buys[signal_ex_col].dropna()
            row[f"avg_signal_excess_fwd_{h}d"] = round(se_vals.mean(), 2) if len(se_vals) > 0 else None
            if len(se_vals) > 0:
                row[f"signal_excess_win_{h}d"] = round((se_vals > 0).mean(), 3)

            # universe excess
            ue_vals = buys[universe_ex_col].dropna()
            row[f"avg_universe_excess_fwd_{h}d"] = round(ue_vals.mean(), 2) if len(ue_vals) > 0 else None
            if len(ue_vals) > 0:
                row[f"universe_excess_win_{h}d"] = round((ue_vals > 0).mean(), 3)

            # entry_price
            ev_vals = buys[entry_col].dropna()
            row[f"avg_entry_fwd_{h}d"] = round(ev_vals.mean(), 2) if len(ev_vals) > 0 else None
            if len(ev_vals) > 0:
                row[f"entry_win_{h}d"] = round((ev_vals > 0).mean(), 3)

        # Stability: avg_universe_excess_fwd_5d / std
        avg_ue5 = row.get("avg_universe_excess_fwd_5d")
        ue_vals = buys["universe_excess_fwd_5d"].dropna()
        std_ue5 = ue_vals.std() if len(ue_vals) > 1 else 0
        if avg_ue5 is not None and std_ue5 > 0:
            row["stability"] = round(avg_ue5 / max(std_ue5, 0.01), 3)
        else:
            row["stability"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Composite score (excess-return focused) ----
    score_weights = {
        "avg_universe_excess_fwd_5d": 0.4,
        "universe_excess_win_5d": 0.3,
        "n_buy_ops": 0.2,
        "stability": 0.1,
    }
    for col in score_weights:
        norm_col = f"{col}_norm"
        if col in df.columns and df[col].notna().any():
            vals = df[col].fillna(0)
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                df[norm_col] = (vals - vmin) / (vmax - vmin)
            else:
                df[norm_col] = 0.0
        else:
            df[norm_col] = 0.0

    df["score"] = (
        df["avg_universe_excess_fwd_5d_norm"].fillna(0) * 0.4 +
        df["universe_excess_win_5d_norm"].fillna(0) * 0.3 +
        df["n_buy_ops_norm"].fillna(0) * 0.2 +
        df["stability_norm"].fillna(0) * 0.1
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Scale OOT with multi-benchmark + entry_price")
    ap.add_argument("--stocks", default="")
    ap.add_argument("--stock-file", default="")
    ap.add_argument("--start-date", default="2025-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--ops-dir", default=str(OPS_DIR))
    ap.add_argument("--output-dir", default=str(OOT_DIR))
    ap.add_argument("--no-prices", action="store_true")
    ap.add_argument("--output-prefix", default="oot_300")
    args = ap.parse_args()

    ops_dir = Path(args.ops_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Step 2: Prices — load ALL from cache for universe benchmark
    print(f"\n{'='*60}")
    print(f"Step 2: Daily OHLC")
    if args.no_prices:
        print("  Loading from cache...")
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
    else:
        prices = prefetch_prices(stocks, args.start_date, args.end_date)

    if not prices:
        print("[ERROR] No price data.")
        sys.exit(1)

    # Step 3: Forward + multi-benchmark + entry_price returns
    print(f"\n{'='*60}")
    print("Step 3: Forward + benchmark + entry_price returns")
    ops = attach_forward_returns(ops, prices)

    for h in FWD_HORIZONS:
        n_valid = ops[f"fwd_{h}d"].notna().sum()
        mean_fwd = ops[f"fwd_{h}d"].dropna().mean()
        mean_win = ops[f"win_{h}d"].dropna().mean()
        mean_signal_ex = ops[f"signal_excess_fwd_{h}d"].dropna().mean()
        mean_universe_ex = ops[f"universe_excess_fwd_{h}d"].dropna().mean()
        n_entry = ops[f"entry_fwd_{h}d"].notna().sum()
        mean_entry = ops[f"entry_fwd_{h}d"].dropna().mean()
        print(f"  fwd_{h}d: {n_valid}/{len(ops)} valid, "
              f"ret={mean_fwd:+.2f}%, win={mean_win:.3f}, "
              f"signal_ex={mean_signal_ex:+.2f}%, universe_ex={mean_universe_ex:+.2f}%")
        if n_entry > 0:
            print(f"         entry_fwd_{h}d: {n_entry}/{len(ops)} valid, ret={mean_entry:+.2f}%")

    # Step 4: Per-stock
    print(f"\n{'='*60}")
    print("Step 4: Per-stock evaluation")
    summary = evaluate_per_stock(ops)

    top_cols = ["stock_code", "n_buy_ops", "avg_fwd_5d", "win_5d",
                "avg_universe_excess_fwd_5d", "universe_excess_win_5d",
                "avg_entry_fwd_5d", "stability", "score"]
    available = [c for c in top_cols if c in summary.columns]
    print(f"\n  Top 10 by score:")
    print(summary[available].head(10).to_string(index=False))

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

    # Distribution
    print(f"\n  Score: mean={summary['score'].mean():.3f}, median={summary['score'].median():.3f}")
    for h in FWD_HORIZONS:
        for m in ["win", "universe_excess_win", "entry_win"]:
            col = f"{m}_{h}d"
            if col in summary.columns:
                vals = summary[col].dropna()
                if len(vals) > 0:
                    print(f"    {col}: mean={vals.mean():.3f}")

    # Missing data
    missing_prices = [s for s in stocks if s not in prices]
    if missing_prices:
        print(f"\n  Stocks missing price data ({len(missing_prices)}): {', '.join(missing_prices[:10])}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
