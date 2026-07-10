"""
v6 cross-day institution tracking pipeline — 300 stocks

Feeds level2_ops (DBSCAN intraday clusters) through InstitutionTracker
per stock, producing cross-day anonymous institutions with Alpha profiles.

Pipeline:
  level2_ops CSV → [per stock] InstitutionTracker → institution registry
  → attach forward returns (batch) → institution-level eval → stock-level summary

用法:
  python scripts/run_v6_pipeline_300.py --stocks 000547,000510,000688  # test subset
  python scripts/run_v6_pipeline_300.py                                # all 300
  python scripts/run_v6_pipeline_300.py --no-prices                    # skip download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.data.price_loader import load_stock_daily
from src.cluster.institution_tracker import InstitutionTracker, extract_fingerprint

FWD_HORIZONS = [1, 3, 5, 10, 20]
OPS_DIR = PROJECT / "data" / "processed" / "level2_ops" / "2025"
STOCKS_PATH = PROJECT / "data" / "processed" / "stock_universe" / "selected_stocks.csv"
DAILY_DIR = PROJECT / "data" / "daily"
OUT_DIR = PROJECT / "data" / "processed" / "v6_institutions"


# ─── Data loading (reused from scale_oot_300.py) ─────────────────────


def load_all_ops(ops_dir: Path, stocks_filter: set[str] | None = None) -> pd.DataFrame:
    files = sorted(ops_dir.glob("level2_ops_*.csv"))
    if not files:
        print(f"[ERROR] No level2_ops_*.csv in {ops_dir}")
        sys.exit(1)

    print(f"Loading {len(files)} ops files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    ops = pd.concat(dfs, ignore_index=True)
    ops["stock_code_clean"] = (
        ops["stock_code"].str.replace(r"\.(SZ|SH)$", "", regex=True).str.zfill(6)
    )
    if stocks_filter:
        ops = ops[ops["stock_code_clean"].isin(stocks_filter)]
    ops["date"] = pd.to_datetime(ops["date"].astype(str), format="%Y%m%d")
    return ops.sort_values(["stock_code_clean", "date"]).reset_index(drop=True)


def prefetch_prices(stocks: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    prices = {}
    n = len(stocks)
    t0 = time.time()
    for i, stock in enumerate(stocks):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Price: {i+1}/{n} ({time.time() - t0:.0f}s)")
        df = load_stock_daily(stock, start_date=start_date, end_date=end_date, adjust="hfq")
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            prices[stock] = df
    print(f"  Price done: {len(prices)}/{n} stocks ({time.time() - t0:.0f}s)")
    return prices


# ─── Per-stock v6 pipeline ────────────────────────────────────────────


def run_v6_for_stock(stock: str, ops: pd.DataFrame,
                     prices: pd.DataFrame | None) -> dict:
    """对单只股票跑 InstitutionTracker，返回机构级评估结果"""
    stock_ops = ops[ops["stock_code_clean"] == stock].copy()
    buys = stock_ops[stock_ops["direction"] == "BUY"].sort_values("date")

    if buys.empty:
        return {"stock": stock, "n_ops": 0, "n_institutions": 0, "institutions": []}

    tracker = InstitutionTracker(match_threshold=0.85)

    # Register all BUY ops chronologically
    for _, row in buys.iterrows():
        op = row.to_dict()
        date_str = row["date"].strftime("%Y%m%d")
        tracker.register_operation(op, date_str, stock)

    records = tracker.get_records_df()
    if records.empty:
        return {"stock": stock, "n_ops": len(buys), "n_institutions": 0, "institutions": []}

    # Attach forward returns (batch, using preloaded prices)
    records = _attach_forward_returns_batch(records, prices, stock)

    # Summarize per institution
    inst_summary = _summarize_institutions(records)
    n_inst = len(inst_summary)

    return {
        "stock": stock,
        "n_ops": len(buys),
        "n_institutions": n_inst,
        "institutions": inst_summary,
    }


def _attach_forward_returns_batch(records: pd.DataFrame,
                                  prices: pd.DataFrame | None,
                                  stock: str) -> pd.DataFrame:
    """批量为机构操作记录附加前向收益（close-to-close hfq）"""
    for h in FWD_HORIZONS:
        records[f"fwd_{h}d"] = np.nan
        records[f"win_{h}d"] = np.nan

    if prices is None or prices.empty:
        return records

    price_dates = prices["date"].dt.strftime("%Y%m%d").values
    price_close = prices["close"].values
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    for idx in records.index:
        op_date = str(records.loc[idx, "date"])
        if op_date not in date_to_idx:
            continue
        di = date_to_idx[op_date]
        close_t = price_close[di]
        if close_t <= 0:
            continue

        for h in FWD_HORIZONS:
            ti = di + h
            if ti < len(price_close):
                ret = (price_close[ti] / close_t - 1) * 100
                records.at[idx, f"fwd_{h}d"] = round(ret, 3)
                records.at[idx, f"win_{h}d"] = 1.0 if ret > 0 else 0.0

    return records


def _summarize_institutions(records: pd.DataFrame) -> list[dict]:
    """按机构聚合，输出每个机构的 Alpha 统计"""
    insts = []
    for inst_id, g in records.groupby("institution_id"):
        n = len(g)
        buy_ops = g[g["direction"] == "BUY"]
        sells = g[g["direction"] == "SELL"]

        summary = {
            "inst_id": inst_id,
            "n_operations": n,
            "n_buy": len(buy_ops),
            "n_sell": len(sells),
            "active_days": g["date"].nunique(),
            "total_buy_wan": round(buy_ops["total_amount_wan"].sum(), 1),
            "total_sell_wan": round(sells["total_amount_wan"].sum(), 1),
            "avg_amount_wan": round(buy_ops["total_amount_wan"].mean(), 1),
            "avg_order_count": round(buy_ops["order_count"].mean(), 1),
            "avg_time_span_min": round(buy_ops["time_span_min"].mean(), 1),
            "net_buy_wan": round(
                buy_ops["total_amount_wan"].sum() - sells["total_amount_wan"].sum(), 1
            ),
        }

        for h in FWD_HORIZONS:
            fwd = buy_ops[f"fwd_{h}d"].dropna()
            wins = buy_ops[f"win_{h}d"].dropna()
            summary[f"avg_fwd_{h}d"] = round(fwd.mean(), 2) if len(fwd) > 0 else None
            summary[f"win_{h}d"] = round(wins.mean(), 3) if len(wins) > 0 else None
            summary[f"n_fwd_{h}d"] = len(fwd)

        # Confidence: based on buy concentration and consistency
        buy_ratio = len(buy_ops) / max(1, n)
        win5 = summary.get("win_5d") or 0
        if buy_ratio >= 0.8 and win5 >= 0.6 and n >= 10:
            summary["confidence"] = "HIGH"
        elif buy_ratio >= 0.6 and win5 >= 0.5 and n >= 5:
            summary["confidence"] = "MEDIUM"
        else:
            summary["confidence"] = "LOW"

        insts.append(summary)

    return sorted(insts, key=lambda x: -(x.get("net_buy_wan") or 0))


# ─── Stock-level aggregation ─────────────────────────────────────────


def build_stock_summary(all_results: list[dict]) -> pd.DataFrame:
    """从各股票的机构汇总中提取股票级指标"""
    rows = []
    for r in all_results:
        insts = r.get("institutions", [])
        high = [i for i in insts if i["confidence"] == "HIGH"]
        med = [i for i in insts if i["confidence"] == "MEDIUM"]
        low = [i for i in insts if i["confidence"] == "LOW"]

        # Pool all institution-level BUY forward returns
        all_fwd = {h: [] for h in FWD_HORIZONS}
        all_wins = {h: [] for h in FWD_HORIZONS}
        for i in insts:
            for h in FWD_HORIZONS:
                if i.get(f"avg_fwd_{h}d") is not None and i.get(f"n_fwd_{h}d", 0) > 0:
                    all_fwd[h].append(i[f"avg_fwd_{h}d"])
                    all_wins[h].append(i[f"win_{h}d"])

        # High-confidence institution stats
        high_fwd5 = [i["avg_fwd_5d"] for i in high if i.get("avg_fwd_5d") is not None]
        high_win5 = [i["win_5d"] for i in high if i.get("win_5d") is not None]

        row = {
            "stock_code": r["stock"],
            "n_ops": r["n_ops"],
            "n_institutions": r["n_institutions"],
            "n_high": len(high),
            "n_medium": len(med),
            "n_low": len(low),
            "low_ratio": round(len(low) / max(1, r["n_institutions"]), 3),
        }
        for h in FWD_HORIZONS:
            row[f"avg_inst_fwd_{h}d"] = round(np.mean(all_fwd[h]), 2) if all_fwd[h] else None
            row[f"avg_inst_win_{h}d"] = round(np.mean(all_wins[h]), 3) if all_wins[h] else None

        row["high_avg_fwd_5d"] = round(np.mean(high_fwd5), 2) if high_fwd5 else None
        row["high_avg_win_5d"] = round(np.mean(high_win5), 3) if high_win5 else None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Composite score
    if "avg_inst_fwd_5d" in df.columns and "avg_inst_win_5d" in df.columns:
        for col in ["avg_inst_win_5d", "avg_inst_fwd_5d", "n_institutions"]:
            vals = df[col].fillna(0)
            vmin, vmax = vals.min(), vals.max()
            df[f"{col}_norm"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0
        df["score"] = (
            df["avg_inst_win_5d_norm"].fillna(0) * 0.35 +
            df["avg_inst_fwd_5d_norm"].fillna(0) * 0.35 +
            df["n_institutions_norm"].fillna(0) * 0.30
        )

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="v6 cross-day institution tracking for 300 stocks")
    ap.add_argument("--stocks", default="",
                    help="Comma-separated stock codes (default: all in selected_stocks.csv)")
    ap.add_argument("--start-date", default="2025-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--ops-dir", default=str(OPS_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-prices", action="store_true")
    args = ap.parse_args()

    ops_dir = Path(args.ops_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stock list
    if args.stocks:
        stocks_filter = set(s.strip().zfill(6) for s in args.stocks.split(","))
    else:
        if STOCKS_PATH.exists():
            sel = pd.read_csv(STOCKS_PATH)
            stocks_filter = set(sel["stock"].astype(str).str.zfill(6))
        else:
            stocks_filter = None

    # Step 1: Load all ops
    print("=" * 60)
    print("Step 1: Load level2_ops")
    t0 = time.time()
    ops = load_all_ops(ops_dir, stocks_filter)
    stocks = sorted(ops["stock_code_clean"].unique())
    print(f"  {len(stocks)} stocks, {len(ops):,} ops ({time.time() - t0:.0f}s)")

    # Step 2: Prefetch prices
    print(f"\n{'='*60}")
    print(f"Step 2: Prefetch prices")
    if args.no_prices:
        prices = {}
        for s in stocks:
            p = DAILY_DIR / f"{s}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                prices[s] = df
        print(f"  Loaded {len(prices)} cached price files")
    else:
        prices = prefetch_prices(stocks, args.start_date, args.end_date)

    # Step 3: Per-stock v6 pipeline
    print(f"\n{'='*60}")
    print(f"Step 3: v6 institution tracking ({len(stocks)} stocks)")
    t0 = time.time()
    all_results = []
    n_inst_total = 0

    for i, stock in enumerate(stocks):
        pdf = prices.get(stock)
        result = run_v6_for_stock(stock, ops, pdf)
        all_results.append(result)
        n_inst_total += result["n_institutions"]

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(stocks)}] {stock}: "
                  f"{result['n_ops']} ops → {result['n_institutions']} insts "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {n_inst_total} total institutions across {len(stocks)} stocks "
          f"({elapsed:.0f}s, {elapsed/len(stocks):.1f}s/stock)")

    # Step 4: Stock-level summary
    print(f"\n{'='*60}")
    print("Step 4: Build stock-level summary")
    summary = build_stock_summary(all_results)

    # Print top/bottom
    print_cols = ["stock_code", "n_ops", "n_institutions", "n_high",
                  "avg_inst_fwd_5d", "avg_inst_win_5d", "high_avg_fwd_5d",
                  "high_avg_win_5d", "score"]
    avail = [c for c in print_cols if c in summary.columns]
    print(f"\n  Top 15 by score:")
    print(summary[avail].head(15).to_string(index=False))

    print(f"\n  Confidence distribution:")
    print(f"    HIGH:   {summary['n_high'].sum()}")
    print(f"    MEDIUM: {summary['n_medium'].sum()}")
    print(f"    LOW:    {summary['n_low'].sum()}")

    # Step 5: Output
    print(f"\n{'='*60}")
    print("Step 5: Output")

    summary_path = out_dir / "v6_stock_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Stock summary: {summary_path}")

    # Per-stock institution details
    inst_dir = out_dir / "institutions"
    inst_dir.mkdir(exist_ok=True)
    for r in all_results:
        if r["institutions"]:
            df = pd.DataFrame(r["institutions"])
            df.to_csv(inst_dir / f"{r['stock']}_institutions.csv", index=False)
    print(f"  Per-stock details: {inst_dir}/ ({len(stocks)} files)")

    # Top institutions across all stocks
    all_insts = []
    for r in all_results:
        stock = r["stock"]
        for inst in r["institutions"]:
            inst["stock_code"] = stock
            all_insts.append(inst)

    if all_insts:
        inst_df = pd.DataFrame(all_insts)
        inst_df = inst_df.sort_values("total_buy_wan", ascending=False)
        inst_path = out_dir / "v6_all_institutions.csv"
        inst_df.to_csv(inst_path, index=False)
        print(f"  All institutions: {inst_path} ({len(inst_df)} rows)")

        # Top HIGH-confidence institutions
        high_insts = inst_df[inst_df["confidence"] == "HIGH"]
        print(f"\n  Top HIGH-confidence institutions ({len(high_insts)}):")
        hcols = ["stock_code", "inst_id", "n_operations", "total_buy_wan",
                 "avg_fwd_5d", "win_5d", "avg_fwd_20d", "win_20d"]
        ha = [c for c in hcols if c in high_insts.columns]
        print(high_insts.nlargest(10, "avg_fwd_5d")[ha].to_string(index=False))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
