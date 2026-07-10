"""
Unified robustness grid using SignalBacktester.

Runs all parameter combinations on 2025 data with the same engine.
Outputs grid summary, all trades, and equity curves.

Grid:
  trend_filter:  none, ma20, ma60
  stop_loss:     None, -0.10, -0.12
  cooldown:      0, 10, 20
  max_positions: 3, 5
  holding_days:  5, 10, 20
  cost_bps:      20 (fixed)
  slippage_bps:  10 (fixed)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtest.signal_backtester import BacktestConfig, SignalBacktester
from src.cluster.institution_tracker import InstitutionTracker

OPS_DIR = PROJECT / "data" / "processed" / "level2_ops" / "2025"
DAILY_DIR = PROJECT / "data" / "daily"
OUT_DIR = PROJECT / "data" / "processed"
PRICE_SCALE = 100

TARGET_STOCKS = ["000547", "000042", "000669", "000688", "000510", "000007", "000426"]

SIGNAL_CONFIG = {
    "min_ops": 10,
    "min_active_days": 3,
    "min_buy_ratio": 0.7,
    "min_net_buy_wan": 0,
    "min_recent_5d_net_buy": 0,
}

GRID = {
    "stock_trend_filter": [None, "ma20", "ma60"],
    "stop_loss": [None, -0.10, -0.12],
    "cooldown_days": [0, 10, 20],
    "max_positions": [3, 5],
    "holding_days": [5, 10, 20],
}


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
            for col in ["open", "high", "low", "close"]:
                df[f"{col}_yuan"] = df[col] / PRICE_SCALE
            prices[s] = df
    return prices


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
    return {"ops_so_far": len(past), "active_days": len(dates),
            "net_buy_wan": round(total_buy, 1)}


def generate_signals(stocks: list[str], ops: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Walk-forward InstitutionTracker — generates signals using T and prior only."""
    all_dates = sorted(set(d for pdf in prices.values() for d in pdf["date_str"].values))
    trackers = {s: InstitutionTracker(match_threshold=0.85) for s in stocks}
    signals = []

    for today in all_dates:
        today_ops = ops[ops["date_str"] == today]
        for stock in stocks:
            stock_ops = today_ops[today_ops["stock_code_clean"] == stock]
            tracker = trackers[stock]
            for _, row in stock_ops.iterrows():
                tracker.register_operation(row.to_dict(), today, stock)

            if not tracker.records:
                continue
            for inst_id, records in tracker.records.items():
                snap = compute_inst_snapshot(records, today)
                if snap is None:
                    continue
                signals.append({
                    "stock_code": stock,
                    "signal_date": today,
                    "inst_id": inst_id,
                    **{f"snap_{k}": v for k, v in snap.items()},
                })

    return pd.DataFrame(signals)


def config_name(params: dict) -> str:
    """Generate compact config name from params."""
    trend = params["stock_trend_filter"] or "none"
    sl = f"sl{abs(int(params['stop_loss']*100))}" if params["stop_loss"] is not None else "slNone"
    cd = f"cd{params['cooldown_days']}"
    mp = f"mp{params['max_positions']}"
    h = f"h{params['holding_days']}"
    return f"{trend}_{sl}_{cd}_{mp}_{h}"


def main():
    print("Loading data...")
    ops = load_ops(TARGET_STOCKS)
    prices = load_prices(TARGET_STOCKS)
    print(f"  {len(ops):,} BUY ops, {len(prices)} stocks")

    print("\nGenerating signals (walk-forward InstitutionTracker)...")
    signals = generate_signals(TARGET_STOCKS, ops, prices)
    print(f"  {len(signals):,} signals")

    # Build all parameter combinations
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = [dict(zip(keys, v)) for v in product(*values)]
    total = len(combos)
    print(f"\nRunning {total} parameter combinations...")

    all_summaries = []
    all_trades = []
    all_equity_curves = []
    progress_interval = max(1, total // 10)

    for i, params in enumerate(combos):
        name = config_name(params)
        cfg = BacktestConfig(
            holding_days=params["holding_days"],
            stop_loss=params["stop_loss"],
            cooldown_days=params["cooldown_days"],
            max_positions=params["max_positions"],
            stock_trend_filter=params["stock_trend_filter"],
            cost_bps=20,
            slippage_bps=10,
            name=name,
        )

        result = SignalBacktester(cfg).run(signals, prices)
        trades_df = result["trades"]
        equity_df = result["equity_curve"]
        summary_df = result["summary"]

        if not trades_df.empty:
            trades_df["config_name"] = name
            all_trades.append(trades_df)

        if not equity_df.empty:
            equity_df["config_name"] = name
            all_equity_curves.append(equity_df)

        if not summary_df.empty:
            for col in ["stock_trend_filter", "stop_loss", "cooldown_days", "max_positions"]:
                summary_df[col] = params[col]
            all_summaries.append(summary_df)

        if (i + 1) % progress_interval == 0:
            print(f"  {i+1}/{total} done")

    # Combine
    grid_df = pd.concat(all_summaries, ignore_index=True)
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_all = pd.concat(all_equity_curves, ignore_index=True) if all_equity_curves else pd.DataFrame()

    # Save
    grid_path = OUT_DIR / "unified_backtest_grid_2025.csv"
    trades_path = OUT_DIR / "unified_backtest_trades_2025.csv"
    equity_path = OUT_DIR / "unified_equity_curve_2025.csv"

    grid_df.to_csv(grid_path, index=False)
    trades_all.to_csv(trades_path, index=False)
    equity_all.to_csv(equity_path, index=False)

    print(f"\nSaved:")
    print(f"  Grid: {grid_path} ({len(grid_df)} rows)")
    print(f"  Trades: {trades_path} ({len(trades_all)} rows)")
    print(f"  Equity: {equity_path} ({len(equity_all)} rows)")

    # ─── Key observations ───
    print(f"\n{'='*70}")
    print("KEY OBSERVATIONS")
    print(f"{'='*70}")

    # 1. Does MA20 kill alpha?
    print("\n1. Stock trend filter impact (5d hold, no stop-loss, no cooldown, max_pos=5):")
    baseline = grid_df[
        (grid_df["holding_days"] == 5) &
        (grid_df["stop_loss"].isna()) &
        (grid_df["cooldown_days"] == 0) &
        (grid_df["max_positions"] == 5)
    ][["config_name", "n_trades", "avg_ret", "win_rate", "total_return", "max_drawdown"]]
    print(baseline.to_string(index=False))

    # 2. Stop-loss comparison
    print("\n2. Stop-loss impact (5d, ma20, cd10, mp5):")
    sl_comp = grid_df[
        (grid_df["holding_days"] == 5) &
        (grid_df["stock_trend_filter"] == "ma20") &
        (grid_df["cooldown_days"] == 10) &
        (grid_df["max_positions"] == 5)
    ][["config_name", "n_trades", "avg_ret", "win_rate", "total_return", "max_drawdown", "stop_loss_exits"]]
    print(sl_comp.to_string(index=False))

    # 3. max_positions comparison
    print("\n3. Max positions impact (5d, trend_none, sl=-0.10, cd10):")
    mp_comp = grid_df[
        (grid_df["holding_days"] == 5) &
        (grid_df["stock_trend_filter"].isna()) &
        (grid_df["stop_loss"] == -0.10) &
        (grid_df["cooldown_days"] == 10)
    ][["config_name", "n_trades", "avg_ret", "win_rate", "total_return", "max_drawdown"]]
    print(mp_comp.to_string(index=False))

    # 4. maxDD sanity check
    print("\n4. Max drawdown sanity:")
    dd_below_neg1 = grid_df[grid_df["max_drawdown"] < -1.0]
    if len(dd_below_neg1) > 0:
        print(f"  FAIL: {len(dd_below_neg1)} configs have maxDD < -100%!")
        print(dd_below_neg1[["config_name", "max_drawdown"]].to_string(index=False))
    else:
        print(f"  PASS: all {len(grid_df)} configs have maxDD >= -100%")
    print(f"  Worst maxDD: {grid_df['max_drawdown'].min():.4%}")
    print(f"  Best maxDD:  {grid_df['max_drawdown'].max():.4%}")

    # 5. Best overall by total_return
    print("\n5. Top 10 by total_return (all holding periods):")
    top10 = grid_df.nlargest(10, "total_return")[
        ["config_name", "holding_days", "n_trades", "avg_ret", "win_rate",
         "total_return", "max_drawdown"]
    ]
    print(top10.to_string(index=False))

    # 6. Best by holding period
    for h in [5, 10, 20]:
        sub = grid_df[grid_df["holding_days"] == h]
        if sub.empty:
            continue
        best = sub.nlargest(3, "total_return")[
            ["config_name", "n_trades", "avg_ret", "win_rate", "total_return", "max_drawdown"]
        ]
        print(f"\n   Top 3 for {h}d hold:")
        print(best.to_string(index=False))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
