"""
Enhanced walk-forward backtest with risk controls.

Adds to backtest_insttracker_signals.py:
  - Market trend filter (index MA20/MA60)
  - Stock trend filter (MA20/MA60)
  - Max past 5-day return cap
  - Stop-loss / take-profit
  - Cooldown after exit
  - Max concurrent positions
  - Configurable cost

Usage:
  python scripts/backtest_enhanced_risk.py --risk-grid
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
PRICE_SCALE = 100

HOLDING_PERIODS = [5, 10, 20]

SIGNAL_CONFIG = {
    "min_ops": 10,
    "min_active_days": 3,
    "min_buy_ratio": 0.7,
    "min_net_buy_wan": 0,
    "min_recent_5d_net_buy": 0,
}

# Index for market filter (中证1000 = 000852)
INDEX_CODE = "000852"


def load_ops(stocks: list[str]) -> pd.DataFrame:
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


def load_index_ma(ma_days: int = 20) -> pd.DataFrame | None:
    """Load index data and compute MA. Returns DataFrame with date_str and ma columns."""
    p = DAILY_DIR / f"{INDEX_CODE}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    df["close_yuan"] = df["close"] / PRICE_SCALE
    df[f"ma{ma_days}"] = df["close_yuan"].rolling(ma_days).mean()
    return df


def compute_inst_snapshot(records: list[dict], as_of_date: str) -> dict | None:
    past = [r for r in records if r["date_str"] <= as_of_date]
    if len(past) < SIGNAL_CONFIG["min_ops"]:
        return None
    dates = sorted(set(r["date_str"] for r in past))
    if len(dates) < SIGNAL_CONFIG["min_active_days"]:
        return None
    total_buy = sum(r.get("total_amount_wan", 0) for r in past)
    if total_buy <= SIGNAL_CONFIG["min_net_buy_wan"]:
        return None
    recent_dates = sorted(dates)[-5:]
    recent_net = sum(r.get("total_amount_wan", 0) for r in past if r["date_str"] in recent_dates)
    if recent_net <= SIGNAL_CONFIG["min_recent_5d_net_buy"]:
        return None
    return {
        "ops_so_far": len(past),
        "active_days": len(dates),
        "buy_ratio": 1.0,
        "net_buy_wan": round(total_buy, 1),
        "recent_5d_net_buy_wan": round(recent_net, 1),
        "first_date": dates[0],
        "last_date": dates[-1],
    }


def check_risk_filters(
    stock: str, today: str, prices: dict, index_df: pd.DataFrame | None,
    cooldown_until: dict[str, str], open_positions: dict,
    risk_params: dict,
) -> tuple[bool, str]:
    """
    Check all risk filters. Returns (pass, reason).
    Filters checked in order; first failure stops.
    """
    pdf = prices.get(stock)
    if pdf is None:
        return False, "no_price_data"

    # 1. Cooldown
    if stock in cooldown_until and today <= cooldown_until[stock]:
        return False, "cooldown"

    # 2. Max positions
    max_pos = risk_params.get("max_positions", 999)
    current_positions = sum(len(v) for v in open_positions.values())
    if current_positions >= max_pos:
        return False, "max_positions"

    # 3. Market filter
    market_filter = risk_params.get("market_filter", "none")
    if market_filter != "none" and index_df is not None:
        ma_col = f"ma{market_filter.replace('ma', '')}"
        idx_row = index_df[index_df["date_str"] == today]
        if not idx_row.empty and ma_col in idx_row.columns:
            if idx_row.iloc[0]["close_yuan"] < idx_row.iloc[0][ma_col]:
                return False, f"market_below_{market_filter}"

    # 4. Stock trend filter
    stock_filter = risk_params.get("stock_trend_filter", "none")
    if stock_filter != "none":
        ma_days = int(stock_filter.replace("ma", ""))
        row = pdf[pdf["date_str"] == today]
        if row.empty:
            return False, "no_stock_price"
        idx = row.index[0]
        if idx < ma_days:
            return False, "insufficient_history"
        ma_val = pdf.iloc[max(0, idx - ma_days + 1):idx + 1]["close_yuan"].mean()
        if pdf.iloc[idx]["close_yuan"] < ma_val:
            return False, f"stock_below_{stock_filter}"

    # 5. Max past 5-day return
    max_past_5d = risk_params.get("max_past_5d_return", 999)
    if max_past_5d < 999:
        row = pdf[pdf["date_str"] == today]
        if not row.empty:
            idx = row.index[0]
            if idx >= 5:
                past_ret = (pdf.iloc[idx]["close_yuan"] / pdf.iloc[idx - 5]["close_yuan"] - 1)
                if past_ret > max_past_5d:
                    return False, f"past_5d_return_{past_ret:.1%}"

    return True, "pass"


def run_backtest(
    stocks: list[str], ops: pd.DataFrame, prices: dict[str, pd.DataFrame],
    risk_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest with risk controls."""

    all_dates = sorted(set(
        d for pdf in prices.values() for d in pdf["date_str"].values
    ))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    index_df = None
    if risk_params.get("market_filter", "none") != "none":
        ma_days = int(risk_params["market_filter"].replace("ma", ""))
        index_df = load_index_ma(ma_days)

    trackers = {s: InstitutionTracker(match_threshold=0.85) for s in stocks}

    open_positions: dict[str, list[dict]] = defaultdict(list)
    cooldown_until: dict[str, str] = {}
    trades: list[dict] = []

    cost_bps = risk_params.get("cost_bps", 10)
    stop_loss = risk_params.get("stop_loss")
    take_profit = risk_params.get("take_profit")
    cooldown_days = risk_params.get("cooldown_days", 0)
    entry_mode = risk_params.get("entry_mode", "open")

    for day_idx, today in enumerate(all_dates):
        # 1. Register today's ops
        today_ops = ops[ops["date_str"] == today]
        for stock in stocks:
            stock_ops = today_ops[today_ops["stock_code_clean"] == stock]
            tracker = trackers[stock]
            for _, row in stock_ops.iterrows():
                tracker.register_operation(row.to_dict(), today, stock)

        # 2. Check stop-loss / take-profit exits (intraday)
        for stock, pos_list in list(open_positions.items()):
            for p in list(pos_list):
                close_today = _get_price(prices, stock, today, "close")
                if close_today is None:
                    continue
                ret_so_far = (close_today / p["entry_price"] - 1)
                exit_reason = None

                if stop_loss is not None and ret_so_far <= stop_loss:
                    exit_reason = "stop_loss"
                elif take_profit is not None and ret_so_far >= take_profit:
                    exit_reason = "take_profit"

                if exit_reason:
                    ret_net = ret_so_far * 100 - cost_bps * 2 / 100
                    p["exit_price"] = close_today
                    p["exit_date"] = today
                    p["return_pct"] = round(ret_so_far * 100, 3)
                    p["return_net_pct"] = round(ret_net, 3)
                    p["exit_reason"] = exit_reason
                    trades.append(p)
                    pos_list.remove(p)
                    if cooldown_days > 0:
                        exit_idx = date_to_idx.get(today, day_idx)
                        cooldown_end_idx = exit_idx + cooldown_days
                        if cooldown_end_idx < len(all_dates):
                            cooldown_until[stock] = all_dates[cooldown_end_idx]

        # 3. Check scheduled exits
        for stock, pos_list in list(open_positions.items()):
            for p in list(pos_list):
                if p["exit_date"] == today:
                    exit_price = _get_price(prices, stock, today, "close")
                    if exit_price is None:
                        continue
                    ret = (exit_price / p["entry_price"] - 1) * 100
                    ret_net = ret - cost_bps * 2 / 100
                    p["exit_price"] = exit_price
                    p["return_pct"] = round(ret, 3)
                    p["return_net_pct"] = round(ret_net, 3)
                    p["exit_reason"] = "maturity"
                    trades.append(p)
                    pos_list.remove(p)
                    if cooldown_days > 0:
                        exit_idx = date_to_idx.get(today, day_idx)
                        cooldown_end_idx = exit_idx + cooldown_days
                        if cooldown_end_idx < len(all_dates):
                            cooldown_until[stock] = all_dates[cooldown_end_idx]

        # 4. Generate new signals
        for stock in stocks:
            if stock in open_positions and len(open_positions[stock]) > 0:
                continue

            # Risk filters
            filter_pass, reason = check_risk_filters(
                stock, today, prices, index_df, cooldown_until, open_positions, risk_params
            )
            if not filter_pass:
                continue

            tracker = trackers[stock]
            if not tracker.records:
                continue

            for inst_id, records in tracker.records.items():
                snap = compute_inst_snapshot(records, today)
                if snap is None:
                    continue

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

    # Close remaining
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
            p["exit_reason"] = "eod"
            trades.append(p)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    # Summary
    summary_rows = []
    for h in HOLDING_PERIODS:
        h_trades = trades_df[trades_df["holding_days"] == h]
        if h_trades.empty:
            continue
        rets = h_trades["return_pct"].dropna()
        rets_net = h_trades["return_net_pct"].dropna()
        cum = rets.cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()

        # Per-stock contribution
        stock_contrib = h_trades.groupby("stock")["return_pct"].sum().sort_values(ascending=False)

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
            "top_stock": stock_contrib.index[0] if len(stock_contrib) > 0 else "",
            "top_stock_pct": round(stock_contrib.iloc[0], 1) if len(stock_contrib) > 0 else 0,
            "stop_loss_exits": int((h_trades["exit_reason"] == "stop_loss").sum()),
            "take_profit_exits": int((h_trades["exit_reason"] == "take_profit").sum()),
        })

    return trades_df, pd.DataFrame(summary_rows)


def _get_price(prices: dict, stock: str, date_str: str, field: str) -> float | None:
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


# ─── Risk Grid ────────────────────────────────────────────────────────


RISK_GRID = [
    {
        "name": "1_baseline",
        "market_filter": "none",
        "stock_trend_filter": "none",
        "max_past_5d_return": 999,
        "stop_loss": None,
        "take_profit": None,
        "cooldown_days": 0,
        "max_positions": 999,
        "cost_bps": 20,
        "entry_mode": "open",
    },
    {
        "name": "2_stock_ma20",
        "market_filter": "none",
        "stock_trend_filter": "ma20",
        "max_past_5d_return": 999,
        "stop_loss": None,
        "take_profit": None,
        "cooldown_days": 0,
        "max_positions": 999,
        "cost_bps": 20,
        "entry_mode": "open",
    },
    {
        "name": "3_ma20+sl10",
        "market_filter": "none",
        "stock_trend_filter": "ma20",
        "max_past_5d_return": 999,
        "stop_loss": -0.10,
        "take_profit": None,
        "cooldown_days": 0,
        "max_positions": 999,
        "cost_bps": 20,
        "entry_mode": "open",
    },
    {
        "name": "4_ma20+sl10+cool20",
        "market_filter": "none",
        "stock_trend_filter": "ma20",
        "max_past_5d_return": 999,
        "stop_loss": -0.10,
        "take_profit": None,
        "cooldown_days": 20,
        "max_positions": 999,
        "cost_bps": 20,
        "entry_mode": "open",
    },
    {
        "name": "5_ma20+sl10+cool20+max5",
        "market_filter": "none",
        "stock_trend_filter": "ma20",
        "max_past_5d_return": 999,
        "stop_loss": -0.10,
        "take_profit": None,
        "cooldown_days": 20,
        "max_positions": 5,
        "cost_bps": 20,
        "entry_mode": "open",
    },
    {
        "name": "6_ma20+past5d30",
        "market_filter": "none",
        "stock_trend_filter": "ma20",
        "max_past_5d_return": 0.30,
        "stop_loss": None,
        "take_profit": None,
        "cooldown_days": 0,
        "max_positions": 999,
        "cost_bps": 20,
        "entry_mode": "open",
    },
]


def main():
    ap = argparse.ArgumentParser(description="Enhanced backtest with risk controls")
    ap.add_argument("--stocks", default="000547,000042,000669,000688,000510,000007,000426")
    ap.add_argument("--risk-grid", action="store_true",
                    help="Run all risk grid combinations")
    ap.add_argument("--entry", default="open", choices=["open", "close"])
    args = ap.parse_args()

    stocks = [s.strip().zfill(6) for s in args.stocks.split(",")]

    print("Loading data...")
    ops = load_ops(stocks)
    prices = load_prices(stocks)
    print(f"  {len(ops):,} BUY ops, {len(prices)} stocks with prices")

    if args.risk_grid:
        print(f"\n{'='*120}")
        print("RISK GRID BACKTEST")
        print(f"{'='*120}")

        all_summaries = []
        for cfg in RISK_GRID:
            print(f"\n--- {cfg['name']} ---")
            cfg["entry_mode"] = args.entry
            trades_df, summary_df = run_backtest(stocks, ops, prices, cfg)

            if summary_df.empty:
                print("  NO TRADES")
                continue

            # Add config to summary
            for k, v in cfg.items():
                if k != "entry_mode":
                    summary_df[k] = v

            h5 = summary_df[summary_df["holding_days"] == 5]
            if not h5.empty:
                r = h5.iloc[0]
                print(f"  5d: {int(r['n_trades'])} trades, avg={r['avg_ret_pct']:.2f}%, "
                      f"win={r['win_rate']:.1%}, total={r['total_return_pct']:.1f}%, "
                      f"dd={r['max_drawdown_pct']:.1f}%, "
                      f"top={r['top_stock']}({r['top_stock_pct']:.0f}%)")

            all_summaries.append(summary_df)

        # Combine and save
        if all_summaries:
            combined = pd.concat(all_summaries, ignore_index=True)
            out_path = OUT_DIR / "insttracker_backtest_risk_grid.csv"
            combined.to_csv(out_path, index=False)
            print(f"\nSaved risk grid: {out_path}")

            # Final grid view (5d only)
            print(f"\n{'='*120}")
            print("RISK GRID SUMMARY (holding 5d)")
            print(f"{'='*120}")
            grid_5d = combined[combined["holding_days"] == 5]
            gcols = ["name", "n_trades", "avg_ret_pct", "win_rate",
                     "total_return_pct", "max_drawdown_pct",
                     "top_stock", "top_stock_pct",
                     "stop_loss_exits", "take_profit_exits"]
            avail = [c for c in gcols if c in grid_5d.columns]
            print(grid_5d[avail].to_string(index=False))
    else:
        # Single run
        cfg = {
            "market_filter": "none",
            "stock_trend_filter": "none",
            "max_past_5d_return": 999,
            "stop_loss": None,
            "take_profit": None,
            "cooldown_days": 0,
            "max_positions": 999,
            "cost_bps": 20,
            "entry_mode": args.entry,
        }
        trades_df, summary_df = run_backtest(stocks, ops, prices, cfg)
        print(f"\n{'='*70}")
        print("BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
