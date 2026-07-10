"""
Signal fusion: merge DBSCAN→InstTracker and v4→v6 daily signals.

Classifies each stock+date into:
  - db_only:      DBSCAN BUY present, v4-v6 no BUY on same day or ±3 days
  - v6_only:      v4-v6 BUY present, DBSCAN no BUY on same day or ±3 days
  - both_confirm: both pipelines have BUY within ±3 days
  - conflict:     DBSCAN BUY present, but v4-v6 has SELL/出货 within ±3 days

Evaluates forward returns for each signal type using L2 daily OHLC.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OPS_DIR = PROJECT / "data" / "processed" / "level2_ops" / "2025"
OOT_DIR = PROJECT / "data" / "processed" / "oot"
OUT_DIR = PROJECT / "data" / "processed"
DAILY_DIR = PROJECT / "data" / "daily"

PRICE_SCALE = 100  # akshare hfq in 分 → 元
FWD_HORIZONS = [5, 10, 20]
CONFIRM_WINDOW = 3  # ±3 days for "both_confirm"


def load_dbscan_signals(stocks: list[str]) -> pd.DataFrame:
    """Load DBSCAN BUY ops, aggregate to stock+date level."""
    files = sorted(OPS_DIR.glob("level2_ops_*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    ops = pd.concat(dfs, ignore_index=True)
    ops["stock_code"] = (
        ops["stock_code"].astype(str).str.replace(r"\.(SZ|SH)$", "", regex=True).str.zfill(6)
    )
    ops = ops[ops["stock_code"].isin(stocks)]
    ops = ops[ops["direction"] == "BUY"]
    ops["date_str"] = ops["date"].astype(str)

    # Aggregate to stock+date: one row per stock per date
    daily = ops.groupby(["stock_code", "date_str"]).agg(
        db_n_clusters=("cluster_id", "count"),
        db_total_buy_wan=("total_amount_wan", "sum"),
        db_avg_price=("avg_price", "mean"),
        db_n_orders=("order_count", "sum"),
    ).reset_index()
    daily["db_signal"] = 1
    return daily


def load_v6_signals(stocks: list[str], confidence_filter: list[str] | None = None) -> pd.DataFrame:
    """Load v4→v6 BUY/SELL operations from crossday_operations_unified.csv."""
    if confidence_filter is None:
        confidence_filter = ["HIGH", "MEDIUM"]

    frames = []
    for stock in stocks:
        f = OOT_DIR / stock / "crossday_operations_unified.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["confidence"].isin(confidence_filter)]
        df["stock_code"] = stock
        df["date_str"] = df["date"].astype(str)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    ops = pd.concat(frames, ignore_index=True)

    # Split BUY vs SELL
    buys = ops[ops["direction"] == "BUY"].groupby(["stock_code", "date_str"]).agg(
        v6_n_buy_ops=("anon_id", "count"),
        v6_total_buy_wan=("amount_wan", "sum"),
        v6_n_buy_insts=("anon_id", "nunique"),
    ).reset_index()
    buys["v6_buy_signal"] = 1

    sells = ops[ops["direction"] == "SELL"].groupby(["stock_code", "date_str"]).agg(
        v6_n_sell_ops=("anon_id", "count"),
        v6_total_sell_wan=("amount_wan", "sum"),
        v6_n_sell_insts=("anon_id", "nunique"),
    ).reset_index()
    sells["v6_sell_signal"] = 1

    # Merge BUY and SELL on stock+date
    daily = buys.merge(sells, on=["stock_code", "date_str"], how="outer").fillna(0)
    for col in ["v6_n_buy_ops", "v6_n_sell_ops", "v6_n_buy_insts", "v6_n_sell_insts"]:
        daily[col] = daily[col].astype(int)
    return daily


def classify_signals(db: pd.DataFrame, v6: pd.DataFrame,
                     stocks: list[str]) -> pd.DataFrame:
    """
    Merge DBSCAN and v6 daily signals, classify into 4 types.

    both_confirm: DBSCAN BUY and v6 BUY within ±CONFIRM_WINDOW days
    db_only:      DBSCAN BUY but no v6 BUY within window
    v6_only:      v6 BUY but no DBSCAN BUY within window
    conflict:     DBSCAN BUY and v6 SELL within ±CONFIRM_WINDOW days
    """
    # Full outer join on stock+date
    merged = db.merge(v6, on=["stock_code", "date_str"], how="outer")
    merged["db_signal"] = merged["db_signal"].fillna(0).astype(int)
    merged["v6_buy_signal"] = merged["v6_buy_signal"].fillna(0).astype(int)
    merged["v6_sell_signal"] = merged["v6_sell_signal"].fillna(0).astype(int)

    # For both_confirm and conflict, check ±window days
    # Build lookup: for each stock, what dates have v6_buy / v6_sell
    v6_buy_dates = defaultdict(set)
    v6_sell_dates = defaultdict(set)
    db_buy_dates = defaultdict(set)

    for _, row in v6.iterrows():
        s, d = row["stock_code"], row["date_str"]
        if row.get("v6_buy_signal"):
            v6_buy_dates[s].add(d)
        if row.get("v6_sell_signal"):
            v6_sell_dates[s].add(d)

    for _, row in db.iterrows():
        db_buy_dates[row["stock_code"]].add(row["date_str"])

    # Get all trading dates
    all_dates = sorted(set(merged["date_str"].dropna().unique()))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    def _nearby(dates_set: set, target: str, window: int) -> bool:
        if target in dates_set:
            return True
        ti = date_to_idx.get(target)
        if ti is None:
            return False
        for w in range(1, window + 1):
            if ti + w < len(all_dates) and all_dates[ti + w] in dates_set:
                return True
            if ti - w >= 0 and all_dates[ti - w] in dates_set:
                return True
        return False

    rows_with_type = []
    for idx, row in merged.iterrows():
        s = row["stock_code"]
        d = row["date_str"]
        has_db = row["db_signal"] == 1
        has_v6_buy = row["v6_buy_signal"] == 1
        has_v6_sell = row["v6_sell_signal"] == 1

        # Check nearby
        nearby_v6_buy = _nearby(v6_buy_dates.get(s, set()), d, CONFIRM_WINDOW)
        nearby_v6_sell = _nearby(v6_sell_dates.get(s, set()), d, CONFIRM_WINDOW)
        nearby_db = _nearby(db_buy_dates.get(s, set()), d, CONFIRM_WINDOW)

        if has_db and nearby_v6_sell and not nearby_v6_buy:
            stype = "conflict"
        elif has_db and nearby_v6_buy:
            stype = "both_confirm"
        elif has_db and not nearby_v6_buy:
            stype = "db_only"
        elif nearby_db and has_v6_buy:
            stype = "both_confirm"
        elif has_v6_buy and not nearby_db:
            stype = "v6_only"
        elif has_db:
            stype = "db_only"
        elif has_v6_buy:
            stype = "v6_only"
        else:
            continue  # no signal on either side — skip

        row_copy = row.to_dict()
        row_copy["signal_type"] = stype
        rows_with_type.append(row_copy)

    result = pd.DataFrame(rows_with_type)
    return result


def attach_forward_returns(signals: pd.DataFrame,
                           prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach close-to-close forward returns for each signal row."""
    for h in FWD_HORIZONS:
        signals[f"fwd_{h}d"] = np.nan
        signals[f"win_{h}d"] = np.nan

    for idx in signals.index:
        stock = signals.at[idx, "stock_code"]
        date_str = signals.at[idx, "date_str"]
        pdf = prices.get(stock)
        if pdf is None:
            continue
        match = pdf[pdf["date_str"] == date_str]
        if match.empty:
            continue
        di = match.index[0]
        close_t = pdf.iloc[di]["close_yuan"]

        for h in FWD_HORIZONS:
            ti = di + h
            if ti < len(pdf):
                ret = (pdf.iloc[ti]["close_yuan"] / close_t - 1) * 100
                signals.at[idx, f"fwd_{h}d"] = round(ret, 3)
                signals.at[idx, f"win_{h}d"] = 1.0 if ret > 0 else 0.0

    return signals


def evaluate_by_type(signals: pd.DataFrame) -> pd.DataFrame:
    """Compute evaluation metrics per signal_type."""
    rows = []
    for stype in ["db_only", "v6_only", "both_confirm", "conflict"]:
        sub = signals[signals["signal_type"] == stype]
        if sub.empty:
            continue

        row = {
            "signal_type": stype,
            "n_signals": len(sub),
            "n_stocks": sub["stock_code"].nunique(),
        }
        for h in FWD_HORIZONS:
            fwd = sub[f"fwd_{h}d"].dropna()
            wins = sub[f"win_{h}d"].dropna()
            row[f"avg_fwd_{h}d"] = round(fwd.mean(), 3) if len(fwd) > 0 else None
            row[f"win_{h}d"] = round(wins.mean(), 3) if len(wins) > 0 else None
            row[f"median_fwd_{h}d"] = round(fwd.median(), 3) if len(fwd) > 0 else None

        # Max drawdown (cumulative)
        fwd5 = sub["fwd_5d"].dropna()
        if len(fwd5) > 0:
            cum = fwd5.cumsum()
            peak = cum.cummax()
            row["max_drawdown"] = round((cum - peak).min(), 3)
        else:
            row["max_drawdown"] = None

        row["avg_holding_days"] = 5  # primary horizon
        rows.append(row)

    return pd.DataFrame(rows)


def load_prices(stocks: list[str]) -> dict[str, pd.DataFrame]:
    """Load daily OHLC from parquet cache."""
    prices = {}
    for s in stocks:
        p = DAILY_DIR / f"{s}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            df["close_yuan"] = df["close"] / PRICE_SCALE
            df["open_yuan"] = df["open"] / PRICE_SCALE
            prices[s] = df
    return prices


def main():
    ap = argparse.ArgumentParser(description="Fusion evaluation of DBSCAN + v4-v6 signals")
    ap.add_argument("--stocks", default="",
                    help="Comma-separated stocks (default: all with v4-v6 OOT data)")
    ap.add_argument("--confidence", default="HIGH,MEDIUM",
                    help="v6 confidence levels to include")
    ap.add_argument("--window", type=int, default=3,
                    help="±N days window for both_confirm")
    args = ap.parse_args()

    global CONFIRM_WINDOW
    CONFIRM_WINDOW = args.window

    conf_filter = [c.strip() for c in args.confidence.split(",")]

    # Determine stocks
    if args.stocks:
        stocks = [s.strip().zfill(6) for s in args.stocks.split(",")]
    else:
        stocks = sorted([
            d.name for d in OOT_DIR.iterdir()
            if d.is_dir() and d.name.isdigit() and
            (d / "crossday_operations_unified.csv").exists()
        ])

    print(f"Stocks with v4-v6 data: {len(stocks)}")
    print(f"v6 confidence filter: {conf_filter}")
    print(f"Confirm window: ±{CONFIRM_WINDOW} days")
    print()

    # Load signals
    print("Loading DBSCAN signals...")
    db = load_dbscan_signals(stocks)
    print(f"  {len(db):,} stock-days with DBSCAN BUY clusters")

    print("Loading v4-v6 signals...")
    v6 = load_v6_signals(stocks, conf_filter)
    print(f"  {len(v6):,} stock-days with v6 operations")

    # Classify
    print("\nClassifying signals...")
    signals = classify_signals(db, v6, stocks)

    # Attach forward returns
    print("Attaching forward returns...")
    prices = load_prices(stocks)
    signals = attach_forward_returns(signals, prices)

    # Evaluate
    print("\nEvaluating by signal type...")
    eval_df = evaluate_by_type(signals)

    # Print
    print(f"\n{'='*100}")
    print("SIGNAL FUSION EVALUATION")
    print(f"{'='*100}")
    fmt_cols = ["signal_type", "n_signals", "n_stocks",
                "avg_fwd_5d", "win_5d", "median_fwd_5d",
                "avg_fwd_10d", "win_10d",
                "avg_fwd_20d", "win_20d", "max_drawdown"]
    avail = [c for c in fmt_cols if c in eval_df.columns]
    print(eval_df[avail].to_string(index=False))

    # Distribution
    print(f"\n{'='*60}")
    print("SIGNAL TYPE DISTRIBUTION")
    print(f"{'='*60}")
    dist = signals["signal_type"].value_counts()
    for stype, cnt in dist.items():
        print(f"  {stype}: {cnt:>6} ({cnt/len(signals)*100:.1f}%)")

    # Save
    signals_path = OUT_DIR / "pipeline_fusion_daily_signals.csv"
    eval_path = OUT_DIR / "pipeline_fusion_eval.csv"

    # Keep essential columns for signals
    sig_cols = ["stock_code", "date_str", "signal_type",
                "db_n_clusters", "db_total_buy_wan",
                "v6_n_buy_ops", "v6_total_buy_wan",
                "v6_n_sell_ops", "v6_total_sell_wan"]
    sig_cols += [f"fwd_{h}d" for h in FWD_HORIZONS]
    sig_cols += [f"win_{h}d" for h in FWD_HORIZONS]
    avail_sig = [c for c in sig_cols if c in signals.columns]
    signals[avail_sig].to_csv(signals_path, index=False)

    eval_df.to_csv(eval_path, index=False)

    print(f"\nSaved:")
    print(f"  Signals: {signals_path} ({len(signals)} rows)")
    print(f"  Evaluation: {eval_path}")

    # Per-stock both_confirm breakdown
    both = signals[signals["signal_type"] == "both_confirm"]
    if not both.empty:
        print(f"\n{'='*60}")
        print("BOTH_CONFIRM per stock (top 10 by count)")
        print(f"{'='*60}")
        per_stock = both.groupby("stock_code").agg(
            n=("signal_type", "count"),
            avg_fwd_5d=("fwd_5d", "mean"),
            win_5d=("win_5d", "mean"),
        ).sort_values("n", ascending=False)
        print(per_stock.head(10).to_string())


if __name__ == "__main__":
    main()
