"""
Walk-forward backtest for DBSCAN + InstitutionTracker signals.

Rules:
  - No forward-looking: all filters use only data available up to day T
  - Entry: T+1 open (or T+1 close), after signal on day T
  - Exit: close at T+N (N = 5, 10, 20 trading days)
  - No duplicate positions: skip signal if stock already has open position

Input:
  - level2_ops CSVs (DBSCAN daily clusters)
  - Daily OHLC (akshare hfq cache in data/daily/)

Output:
  - data/processed/insttracker_backtest_trades.csv
  - data/processed/insttracker_backtest_summary.csv
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

from src.cluster.institution_tracker import InstitutionTracker

OPS_DIR = PROJECT / "data" / "processed" / "level2_ops" / "2025"
DAILY_DIR = PROJECT / "data" / "daily"
OUT_DIR = PROJECT / "data" / "processed"
PRICE_SCALE = 100  # akshare hfq prices are in 分, divide by 100 for 元

HOLDING_PERIODS = [5, 10, 20]

# Signal criteria (past-only, checked at end of day T)
SIGNAL_CONFIG = {
    "min_ops": 10,
    "min_active_days": 3,
    "min_buy_ratio": 0.7,
    "min_net_buy_wan": 0,       # net_buy_wan > 0
    "min_recent_5d_net_buy": 0,  # recent 5-day net buy > 0
}


def load_ops(stocks: list[str]) -> pd.DataFrame:
    """Load level2_ops for target stocks, BUY direction only."""
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
    ops["stock_code_clean"] = (
        ops["stock_code"].astype(str).str.replace(r"\.(SZ|SH)$", "", regex=True).str.zfill(6)
    )
    ops = ops[ops["stock_code_clean"].isin(stocks)]
    ops = ops[ops["direction"] == "BUY"]
    ops["date_str"] = pd.to_datetime(ops["date"].astype(str), format="%Y%m%d").dt.strftime("%Y%m%d")
    return ops.sort_values(["stock_code_clean", "date_str"]).reset_index(drop=True)


def load_prices(stocks: list[str]) -> dict[str, pd.DataFrame]:
    """Load daily OHLC from parquet cache, return close series in 元."""
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


def compute_inst_snapshot(records: list[dict], as_of_date: str) -> dict | None:
    """
    Compute cumulative stats for an institution using only records up to as_of_date.
    Returns None if min criteria not met.
    """
    past = [r for r in records if r["date_str"] <= as_of_date]
    if len(past) < SIGNAL_CONFIG["min_ops"]:
        return None

    dates = sorted(set(r["date_str"] for r in past))
    if len(dates) < SIGNAL_CONFIG["min_active_days"]:
        return None

    buy_ops = past  # all are BUY ops in our setup
    total_buy = sum(r.get("total_amount_wan", 0) for r in buy_ops)
    if total_buy <= 0:
        return None

    # buy_ratio is 1.0 for pure buy institutions, but keep for generality
    buy_ratio = 1.0
    if buy_ratio < SIGNAL_CONFIG["min_buy_ratio"]:
        return None

    # Recent 5-day net buy
    recent_dates = sorted(dates)[-5:]
    recent_net = sum(
        r.get("total_amount_wan", 0) for r in past if r["date_str"] in recent_dates
    )
    if recent_net <= SIGNAL_CONFIG["min_recent_5d_net_buy"]:
        return None

    net_buy = total_buy
    if net_buy <= SIGNAL_CONFIG["min_net_buy_wan"]:
        return None

    return {
        "ops_so_far": len(past),
        "active_days": len(dates),
        "buy_ratio": round(buy_ratio, 3),
        "net_buy_wan": round(net_buy, 1),
        "recent_5d_net_buy_wan": round(recent_net, 1),
        "first_date": dates[0],
        "last_date": dates[-1],
    }


def run_backtest(
    stocks: list[str],
    ops: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    entry_mode: str = "open",  # "open" or "close"
    cost_bps: float = 10.0,     # 10 bps per side (印花税+佣金)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest.

    For each trading day:
      1. Register day's ops through InstitutionTracker
      2. Compute institution snapshots
      3. Check signal criteria
      4. Generate entry signals for next trading day
      5. Track open positions and exit at holding period
    """
    # Get all trading days from price data
    all_dates = sorted(set(
        d for pdf in prices.values()
        for d in pdf["date_str"].values
    ))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Per-stock InstitutionTracker
    trackers: dict[str, InstitutionTracker] = {
        s: InstitutionTracker(match_threshold=0.85) for s in stocks
    }

    # Position tracking: list of (stock, entry_date, entry_price, exit_date, holding_period)
    positions: list[dict] = []
    open_positions: dict[str, list[dict]] = defaultdict(list)  # stock → active positions
    trades: list[dict] = []

    # Process day by day
    for day_idx, today in enumerate(all_dates):
        # 1. Register today's ops
        today_ops = ops[ops["date_str"] == today]
        for stock in stocks:
            stock_ops = today_ops[today_ops["stock_code_clean"] == stock]
            tracker = trackers[stock]
            for _, row in stock_ops.iterrows():
                op = row.to_dict()
                tracker.register_operation(op, today, stock)

        # 2. Check for exits (positions that mature today)
        newly_closed = []
        for pos in open_positions.values():
            for p in list(pos):
                if p["exit_date"] == today:
                    exit_price = _get_price(prices, p["stock"], today, "close")
                    if exit_price is None:
                        continue
                    ret = (exit_price / p["entry_price"] - 1) * 100
                    # Apply costs (entry + exit)
                    ret_net = ret - cost_bps * 2 / 100
                    p["exit_price"] = exit_price
                    p["return_pct"] = round(ret, 3)
                    p["return_net_pct"] = round(ret_net, 3)
                    trades.append(p)
                    pos.remove(p)

        # 3. Check signals from current institution snapshots
        for stock in stocks:
            # Skip if stock already has open position
            if stock in open_positions and len(open_positions[stock]) > 0:
                continue

            tracker = trackers[stock]
            if not tracker.records:
                continue

            for inst_id, records in tracker.records.items():
                snap = compute_inst_snapshot(records, today)
                if snap is None:
                    continue

                # SIGNAL FIRED — enter at next trading day
                next_day_idx = day_idx + 1
                if next_day_idx >= len(all_dates):
                    continue
                next_date = all_dates[next_day_idx]

                entry_price = _get_price(prices, stock, next_date, entry_mode)
                if entry_price is None or entry_price <= 0:
                    continue

                for h in HOLDING_PERIODS:
                    exit_idx = next_day_idx + h
                    if exit_idx >= len(all_dates):
                        continue
                    exit_date = all_dates[exit_idx]

                    pos = {
                        "stock": stock,
                        "inst_id": inst_id,
                        "signal_date": today,
                        "entry_date": next_date,
                        "entry_price": round(entry_price, 4),
                        "exit_date": exit_date,
                        "holding_days": h,
                        "entry_mode": entry_mode,
                        **{f"snap_{k}": v for k, v in snap.items()},
                    }
                    open_positions[stock].append(pos)

    # Close any remaining open positions at last available price
    last_date = all_dates[-1]
    for stock, pos_list in open_positions.items():
        for p in list(pos_list):
            exit_price = _get_price(prices, stock, last_date, "close")
            if exit_price is None:
                continue
            ret = (exit_price / p["entry_price"] - 1) * 100
            ret_net = ret - cost_bps * 2 / 100
            p["exit_price"] = exit_price
            p["return_pct"] = round(ret, 3)
            p["return_net_pct"] = round(ret_net, 3)
            trades.append(p)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    # Build summary
    summary_rows = []
    for h in HOLDING_PERIODS:
        h_trades = trades_df[trades_df["holding_days"] == h]
        if h_trades.empty:
            continue
        rets = h_trades["return_pct"].dropna()
        rets_net = h_trades["return_net_pct"].dropna()

        # Drawdown: cumulative P&L
        cum = rets.cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()

        summary_rows.append({
            "holding_days": h,
            "n_trades": len(h_trades),
            "n_stocks": h_trades["stock"].nunique(),
            "avg_ret_pct": round(rets.mean(), 3),
            "win_rate": round((rets > 0).mean(), 3),
            "avg_ret_net_pct": round(rets_net.mean(), 3),
            "max_drawdown_pct": round(dd, 3),
            "total_return_pct": round(rets.sum(), 3),
            "total_return_net_pct": round(rets_net.sum(), 3),
            "best_trade_pct": round(rets.max(), 3),
            "worst_trade_pct": round(rets.min(), 3),
            "cost_assumption": f"{cost_bps}bps per side",
            "entry_mode": entry_mode,
        })

    summary_df = pd.DataFrame(summary_rows)
    return trades_df, summary_df


def _get_price(prices: dict, stock: str, date_str: str, field: str) -> float | None:
    """Get price for a stock on a given date. field = 'open' or 'close'."""
    pdf = prices.get(stock)
    if pdf is None:
        return None
    row = pdf[pdf["date_str"] == date_str]
    if row.empty:
        return None
    col = f"{field}_yuan"
    if col not in row.columns:
        return None
    return float(row.iloc[0][col])


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest for InstTracker signals")
    ap.add_argument("--stocks", default="000547,000042,000669,000688,000510,000007,000426",
                    help="Comma-separated stock codes")
    ap.add_argument("--entry", default="open", choices=["open", "close"],
                    help="Entry price: T+1 open or T+1 close")
    ap.add_argument("--cost", type=float, default=10.0,
                    help="Transaction cost in bps per side (default 10 = 0.1%%)")
    args = ap.parse_args()

    stocks = [s.strip().zfill(6) for s in args.stocks.split(",")]
    print(f"Stocks: {stocks}")
    print(f"Entry: T+1 {args.entry}, Cost: {args.cost} bps/side")
    print(f"Signal criteria: {SIGNAL_CONFIG}")
    print()

    # Load data
    print("Loading ops...")
    ops = load_ops(stocks)
    print(f"  {len(ops):,} BUY ops across {ops['stock_code_clean'].nunique()} stocks")

    print("Loading prices...")
    prices = load_prices(stocks)
    print(f"  {len(prices)} stocks with price data")

    # Run backtest
    print(f"\nRunning walk-forward backtest...")
    trades_df, summary_df = run_backtest(
        stocks, ops, prices,
        entry_mode=args.entry,
        cost_bps=args.cost,
    )

    if trades_df.empty:
        print("No trades generated!")
        return

    # Output
    trades_path = OUT_DIR / "insttracker_backtest_trades.csv"
    summary_path = OUT_DIR / "insttracker_backtest_summary.csv"
    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print("BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\nTrades: {trades_path} ({len(trades_df)} rows)")
    print(f"Summary: {summary_path}")

    # Per-stock breakdown
    print(f"\n{'='*70}")
    print("PER-STOCK (holding 5d)")
    print(f"{'='*70}")
    h5 = trades_df[trades_df["holding_days"] == 5]
    for stock in stocks:
        st = h5[h5["stock"] == stock]
        if st.empty:
            continue
        rets = st["return_pct"]
        print(f"  {stock}: {len(st):>3} trades, avg_ret={rets.mean():.2f}%, "
              f"win={(rets>0).mean():.1%}, total={rets.sum():.1f}%")


if __name__ == "__main__":
    main()
