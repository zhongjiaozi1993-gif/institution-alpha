"""
Robustness validation suite:
  - Monthly return stability report
  - Trading feasibility filters (limit-up, suspension, liquidity)
  - Parameter sensitivity analysis
  - Yearly OOT stub (requires level2_ops for other years)

Usage:
  python scripts/robustness_validation.py --all
"""

from __future__ import annotations

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

TARGET_STOCKS = ["000547", "000042", "000669", "000688", "000510", "000007", "000426"]

BASE_RISK = {
    "stock_trend_filter": "ma20",
    "stop_loss": -0.10,
    "cooldown_days": 20,
    "max_positions": 5,
    "cost_bps": 20,
    "entry_mode": "open",
    "market_filter": "none",
    "max_past_5d_return": 999,
    "take_profit": None,
}

HOLDING_PERIODS = [5, 10, 20]

SIGNAL_CONFIG = {
    "min_ops": 10,
    "min_active_days": 3,
    "min_buy_ratio": 0.7,
    "min_net_buy_wan": 0,
    "min_recent_5d_net_buy": 0,
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
            df["close_yuan"] = df["close"] / PRICE_SCALE
            df["open_yuan"] = df["open"] / PRICE_SCALE
            if "high" in df.columns:
                df["high_yuan"] = df["high"] / PRICE_SCALE
            if "amount" in df.columns:
                df["amount_yuan"] = df["amount"]  # already in yuan, not 分
            elif "volume" in df.columns:
                df["amount_yuan"] = df["volume"]  # fallback: volume in shares
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
    return {
        "ops_so_far": len(past),
        "active_days": len(dates),
        "net_buy_wan": round(total_buy, 1),
        "recent_5d_net_buy_wan": round(recent_net, 1),
    }


# ─── Trading feasibility checks ───────────────────────────────────────


def is_limit_up(stock: str, date_str: str, prices: dict) -> bool:
    """Check if stock hit +10% limit up on given date (close ≈ high ≈ prev_close * 1.10)."""
    pdf = prices.get(stock)
    if pdf is None:
        return False
    row = pdf[pdf["date_str"] == date_str]
    if row.empty:
        return False
    idx = row.index[0]
    if idx < 1:
        return False
    prev_close = pdf.iloc[idx - 1]["close_yuan"]
    today_close = row.iloc[0]["close_yuan"]
    if prev_close <= 0:
        return False
    chg = (today_close / prev_close - 1) * 100
    # Check if change >= 9.8% (close to 10% limit, allowing for rounding)
    if chg >= 9.8:
        # Also check if high == low (one-word board) or close == high
        if "high_yuan" in row.columns:
            if row.iloc[0]["close_yuan"] >= row.iloc[0]["high_yuan"] * 0.995:
                return True
        else:
            return True
    return False


def is_suspended(stock: str, date_str: str, prices: dict) -> bool:
    """Rough check: no price data for this date (trading halt)."""
    pdf = prices.get(stock)
    if pdf is None:
        return True
    row = pdf[pdf["date_str"] == date_str]
    return row.empty


def get_liquidity(stock: str, date_str: str, prices: dict) -> float:
    """Get daily turnover amount in 万元."""
    pdf = prices.get(stock)
    if pdf is None:
        return 0
    row = pdf[pdf["date_str"] == date_str]
    if row.empty or "amount_yuan" not in row.columns:
        return 0
    return float(row.iloc[0]["amount_yuan"])


# ─── Enhanced backtest with tradeability ──────────────────────────────


def run_backtest_tradeable(
    stocks: list[str], ops: pd.DataFrame, prices: dict,
    risk_params: dict, tradeability: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Walk-forward backtest with risk controls AND trading feasibility.
    Returns (trades_df, summary_df, feasibility_stats).
    """
    all_dates = sorted(set(d for pdf in prices.values() for d in pdf["date_str"].values))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    trackers = {s: InstitutionTracker(match_threshold=0.85) for s in stocks}
    open_positions: dict[str, list[dict]] = defaultdict(list)
    cooldown_until: dict[str, str] = {}
    trades: list[dict] = []

    # Feasibility counters
    fstat = {
        "n_signals": 0, "n_executable": 0,
        "skipped_limit_up": 0, "skipped_suspended": 0,
        "skipped_low_liquidity": 0, "skipped_low_amount": 0,
    }

    cost_bps = risk_params.get("cost_bps", 20)
    stop_loss = risk_params.get("stop_loss")
    cooldown_days = risk_params.get("cooldown_days", 0)
    max_pos = risk_params.get("max_positions", 999)
    entry_mode = risk_params.get("entry_mode", "open")
    min_amount_wan = tradeability.get("min_amount_wan", 1000)  # 1000万成交额
    max_pct_of_volume = tradeability.get("max_pct_of_volume", 0.03)  # 3%
    slippage_bps = tradeability.get("slippage_bps", 10)

    for day_idx, today in enumerate(all_dates):
        # Register ops
        today_ops = ops[ops["date_str"] == today]
        for stock in stocks:
            stock_ops = today_ops[today_ops["stock_code_clean"] == stock]
            tracker = trackers[stock]
            for _, row in stock_ops.iterrows():
                tracker.register_operation(row.to_dict(), today, stock)

        # Check exits (stop-loss first, then scheduled)
        for stock, pos_list in list(open_positions.items()):
            for p in list(pos_list):
                close_today = _get_price(prices, stock, today, "close")
                if close_today is None:
                    continue
                ret_so_far = (close_today / p["entry_price"] - 1)
                exit_reason = None

                if stop_loss is not None and ret_so_far <= stop_loss:
                    exit_reason = "stop_loss"
                elif p["exit_date"] == today:
                    exit_reason = "maturity"

                if exit_reason:
                    ret_net = ret_so_far * 100 - cost_bps * 2 / 100 - slippage_bps * 2 / 10000
                    p["exit_price"] = close_today
                    p["exit_date_actual"] = today
                    p["return_pct"] = round(ret_so_far * 100, 3)
                    p["return_net_pct"] = round(ret_net, 3)
                    p["exit_reason"] = exit_reason
                    trades.append(p)
                    pos_list.remove(p)
                    if cooldown_days > 0:
                        exit_idx = date_to_idx.get(today, day_idx)
                        cd_end = exit_idx + cooldown_days
                        if cd_end < len(all_dates):
                            cooldown_until[stock] = all_dates[cd_end]

        # Generate signals
        for stock in stocks:
            if stock in open_positions and len(open_positions[stock]) > 0:
                continue
            if stock in cooldown_until and today <= cooldown_until[stock]:
                continue

            current_positions = sum(len(v) for v in open_positions.values())
            if current_positions >= max_pos:
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
                fstat["n_signals"] += 1

                # ─── Tradeability checks ───
                entry_ok = True

                if tradeability.get("check_limit_up", True):
                    if is_limit_up(stock, next_date, prices):
                        fstat["skipped_limit_up"] += 1
                        entry_ok = False

                if tradeability.get("check_suspended", True):
                    if is_suspended(stock, next_date, prices):
                        fstat["skipped_suspended"] += 1
                        entry_ok = False

                if tradeability.get("check_liquidity", True):
                    liq = get_liquidity(stock, next_date, prices)
                    liq_wan = liq / 10000  # yuan → 万元
                    if liq_wan < min_amount_wan:
                        fstat["skipped_low_liquidity"] += 1
                        entry_ok = False

                # Stock trend filter: close > MA on signal date
                trend_filter = risk_params.get("stock_trend_filter", "none")
                if trend_filter != "none" and trend_filter is not None:
                    ma_window = 20 if trend_filter == "ma20" else 60
                    pdf = prices.get(stock)
                    if pdf is not None:
                        today_idx = pdf[pdf["date_str"] == today].index
                        if len(today_idx) > 0:
                            ti = today_idx[0]
                            if ti >= ma_window:
                                ma = pdf.iloc[ti - ma_window + 1:ti + 1]["close_yuan"].mean()
                                close_t = pdf.iloc[ti]["close_yuan"]
                                if close_t <= ma:
                                    entry_ok = False

                if not entry_ok:
                    continue

                fstat["n_executable"] += 1
                entry_price = _get_price(prices, stock, next_date, entry_mode)
                if entry_price is None or entry_price <= 0:
                    continue

                # Apply slippage to entry
                entry_price *= (1 + slippage_bps / 10000)

                for h in HOLDING_PERIODS:
                    exit_idx = next_day_idx + h
                    if exit_idx >= len(all_dates):
                        continue
                    exit_date = all_dates[exit_idx]
                    pos = {
                        "stock": stock, "inst_id": inst_id,
                        "signal_date": today, "entry_date": next_date,
                        "entry_price": round(entry_price, 4),
                        "exit_date": exit_date, "holding_days": h,
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
            ret_net = ret - cost_bps * 2 / 100 - slippage_bps * 2 / 10000
            p["exit_price"] = exit_price
            p["return_pct"] = round(ret, 3)
            p["return_net_pct"] = round(ret_net, 3)
            p["exit_reason"] = "eod"
            trades.append(p)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame(), fstat

    # Summary
    summary_rows = []
    for h in HOLDING_PERIODS:
        h_trades = trades_df[trades_df["holding_days"] == h]
        if h_trades.empty:
            continue
        rets = h_trades["return_pct"].dropna()
        cum = rets.cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()
        stock_contrib = h_trades.groupby("stock")["return_pct"].sum().sort_values(ascending=False)

        # Monthly breakdown
        h_trades["year_month"] = pd.to_datetime(h_trades["entry_date"]).dt.strftime("%Y-%m")
        monthly = h_trades.groupby("year_month")["return_pct"].agg(["count", "mean", "sum"])
        positive_months = (monthly["sum"] > 0).sum()

        summary_rows.append({
            "holding_days": h,
            "n_trades": len(h_trades),
            "n_stocks": h_trades["stock"].nunique(),
            "avg_ret_pct": round(rets.mean(), 3),
            "win_rate": round((rets > 0).mean(), 3),
            "total_return_pct": round(rets.sum(), 3),
            "max_drawdown_pct": round(dd, 3),
            "top_stock": stock_contrib.index[0] if len(stock_contrib) > 0 else "",
            "top_stock_pct": round(stock_contrib.iloc[0], 1) if len(stock_contrib) > 0 else 0,
            "worst_stock": stock_contrib.index[-1] if len(stock_contrib) > 0 else "",
            "n_months": len(monthly),
            "positive_months": positive_months,
            "monthly_positive_rate": round(positive_months / max(1, len(monthly)), 3),
        })

    return trades_df, pd.DataFrame(summary_rows), fstat


# ─── Monthly Report ────────────────────────────────────────────────────


def monthly_report(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly return breakdown."""
    if trades_df.empty:
        return pd.DataFrame()

    t5 = trades_df[trades_df["holding_days"] == 5].copy()
    t5["year_month"] = pd.to_datetime(t5["entry_date"]).dt.strftime("%Y-%m")

    rows = []
    for ym in sorted(t5["year_month"].unique()):
        m = t5[t5["year_month"] == ym]
        rets = m["return_pct"]
        if len(rets) == 0:
            continue
        cum = rets.cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()
        top_stock = m.groupby("stock")["return_pct"].sum().nlargest(1)

        rows.append({
            "year_month": ym,
            "n_trades": len(m),
            "avg_ret": round(rets.mean(), 3),
            "win_rate": round((rets > 0).mean(), 3),
            "total_return": round(rets.sum(), 3),
            "max_drawdown": round(dd, 3),
            "top_stock": top_stock.index[0] if len(top_stock) > 0 else "",
            "top_stock_contribution": round(top_stock.iloc[0], 1) if len(top_stock) > 0 else 0,
        })

    return pd.DataFrame(rows)


# ─── Parameter Sensitivity ─────────────────────────────────────────────


SENSITIVITY_GRID = []
for sl in [-0.08, -0.10, -0.12]:
    for cd in [10, 20, 30]:
        for mp in [3, 5, 8]:
            SENSITIVITY_GRID.append({
                "name": f"sl{int(abs(sl)*100)}_cd{cd}_mp{mp}",
                "stop_loss": sl, "cooldown_days": cd, "max_positions": mp,
            })

STOCK_TREND_SENS = [
    {"name": "trend_none", "stock_trend_filter": "none"},
    {"name": "trend_ma20", "stock_trend_filter": "ma20"},
    {"name": "trend_ma60", "stock_trend_filter": "ma60"},
]


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


# ─── Main ──────────────────────────────────────────────────────────────


def main():
    print("Loading data...")
    ops = load_ops(TARGET_STOCKS)
    prices = load_prices(TARGET_STOCKS)
    print(f"  {len(ops):,} BUY ops, {len(prices)} stocks")

    # ── Task 2: Monthly Report ──
    print(f"\n{'='*70}")
    print("TASK 2: Monthly Return Stability Report")
    print(f"{'='*70}")

    # Use base risk params with tradeability
    trade_cfg = {
        "check_limit_up": True, "check_suspended": True,
        "check_liquidity": True, "min_amount_wan": 1000,
        "max_pct_of_volume": 0.03, "slippage_bps": 10,
    }
    trades_df, summary_df, fstat = run_backtest_tradeable(
        TARGET_STOCKS, ops, prices, BASE_RISK, trade_cfg
    )
    monthly = monthly_report(trades_df)

    print(f"\nFeasibility stats:")
    for k, v in fstat.items():
        print(f"  {k}: {v}")
    print(f"  execution_rate: {fstat['n_executable']/max(1,fstat['n_signals'])*100:.1f}%")

    print(f"\nMonthly breakdown (5d hold):")
    print(monthly.to_string(index=False))

    monthly_path = OUT_DIR / "monthly_return_report.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"\nSaved: {monthly_path}")

    # ── Task 3: Tradeability Comparison ──
    print(f"\n{'='*70}")
    print("TASK 3: Trading Feasibility Impact")
    print(f"{'='*70}")

    trade_configs = [
        ("no_filters", {"check_limit_up": False, "check_suspended": False,
                        "check_liquidity": False, "slippage_bps": 0}),
        ("limit_up_only", {"check_limit_up": True, "check_suspended": False,
                           "check_liquidity": False, "slippage_bps": 10}),
        ("all_filters_1pct", {"check_limit_up": True, "check_suspended": True,
                              "check_liquidity": True, "min_amount_wan": 1000,
                              "max_pct_of_volume": 0.01, "slippage_bps": 10}),
        ("all_filters_3pct", {"check_limit_up": True, "check_suspended": True,
                              "check_liquidity": True, "min_amount_wan": 1000,
                              "max_pct_of_volume": 0.03, "slippage_bps": 10}),
        ("all_filters_slip20", {"check_limit_up": True, "check_suspended": True,
                                "check_liquidity": True, "min_amount_wan": 1000,
                                "max_pct_of_volume": 0.03, "slippage_bps": 20}),
    ]

    tradeability_rows = []
    for tname, tcfg in trade_configs:
        _, tsum, tfstat = run_backtest_tradeable(
            TARGET_STOCKS, ops, prices, BASE_RISK, tcfg
        )
        if tsum.empty:
            continue
        h5 = tsum[tsum["holding_days"] == 5]
        if not h5.empty:
            r = h5.iloc[0]
            tradeability_rows.append({
                "config": tname,
                "n_signals": tfstat["n_signals"],
                "n_executable_trades": tfstat["n_executable"],
                "skipped_limit_up": tfstat["skipped_limit_up"],
                "skipped_suspended": tfstat["skipped_suspended"],
                "skipped_low_liquidity": tfstat["skipped_low_liquidity"],
                "n_trades": int(r["n_trades"]),
                "avg_ret": r["avg_ret_pct"],
                "win_rate": r["win_rate"],
                "total_return": r["total_return_pct"],
                "max_drawdown": r["max_drawdown_pct"],
            })

    ta_df = pd.DataFrame(tradeability_rows)
    print(ta_df.to_string(index=False))
    ta_path = OUT_DIR / "backtest_tradeability_summary.csv"
    ta_df.to_csv(ta_path, index=False)
    print(f"\nSaved: {ta_path}")

    # ── Task 4: Parameter Sensitivity ──
    print(f"\n{'='*70}")
    print("TASK 4: Parameter Sensitivity Analysis")
    print(f"{'='*70}")

    # Stock trend sensitivity (fast)
    print("\nStock trend filter sensitivity:")
    trend_rows = []
    for tcfg in STOCK_TREND_SENS:
        risk = {**BASE_RISK, **tcfg}
        _, tsum, _ = run_backtest_tradeable(
            TARGET_STOCKS, ops, prices, risk, trade_cfg
        )
        if tsum.empty:
            continue
        h5 = tsum[tsum["holding_days"] == 5]
        if not h5.empty:
            r = h5.iloc[0]
            print(f"  {tcfg['name']}: {int(r['n_trades'])} trades, "
                  f"avg={r['avg_ret_pct']:.2f}%, win={r['win_rate']:.1%}, "
                  f"dd={r['max_drawdown_pct']:.1f}%, total={r['total_return_pct']:.1f}%")
            trend_rows.append({
                "param": "stock_trend_filter", "value": tcfg["stock_trend_filter"],
                "n_trades": int(r["n_trades"]), "avg_ret": r["avg_ret_pct"],
                "win_rate": r["win_rate"], "total_return": r["total_return_pct"],
                "max_drawdown": r["max_drawdown_pct"],
            })

    # Stop-loss + cooldown + max_positions grid
    print(f"\nStop-loss × cooldown × max_positions grid ({len(SENSITIVITY_GRID)} combos):")
    sens_rows = trend_rows.copy()
    for i, scfg in enumerate(SENSITIVITY_GRID):
        risk = {
            **BASE_RISK,
            "stop_loss": scfg["stop_loss"],
            "cooldown_days": scfg["cooldown_days"],
            "max_positions": scfg["max_positions"],
        }
        _, tsum, _ = run_backtest_tradeable(
            TARGET_STOCKS, ops, prices, risk, trade_cfg
        )
        if tsum.empty:
            continue
        h5 = tsum[tsum["holding_days"] == 5]
        if not h5.empty:
            r = h5.iloc[0]
            sens_rows.append({
                "param": "combo", "value": scfg["name"],
                "stop_loss": scfg["stop_loss"],
                "cooldown_days": scfg["cooldown_days"],
                "max_positions": scfg["max_positions"],
                "n_trades": int(r["n_trades"]), "avg_ret": r["avg_ret_pct"],
                "win_rate": r["win_rate"], "total_return": r["total_return_pct"],
                "max_drawdown": r["max_drawdown_pct"],
            })
        if (i + 1) % 9 == 0:
            print(f"  {i+1}/{len(SENSITIVITY_GRID)} done")

    sens_df = pd.DataFrame(sens_rows)
    sens_path = OUT_DIR / "risk_param_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)

    # Show top 5 combos by a simple quality metric (high return + low drawdown)
    if "avg_ret" in sens_df.columns and "max_drawdown" in sens_df.columns:
        combo = sens_df[sens_df["param"] == "combo"].copy()
        combo["quality"] = combo["avg_ret"] + combo["max_drawdown"] * 0.5  # reward ret, penalize dd
        top5 = combo.nlargest(5, "quality")
        print(f"\nTop 5 parameter combos (by avg_ret - 0.5*|dd|):")
        print(top5[["value", "n_trades", "avg_ret", "win_rate", "max_drawdown", "total_return"]].to_string(index=False))

        print(f"\nWorst 3 combos:")
        worst3 = combo.nsmallest(3, "quality")
        print(worst3[["value", "n_trades", "avg_ret", "win_rate", "max_drawdown", "total_return"]].to_string(index=False))

    print(f"\nSaved: {sens_path}")

    # ── Task 1: Yearly OOT (stub) ──
    print(f"\n{'='*70}")
    print("TASK 1: Yearly OOT (stub)")
    print(f"{'='*70}")
    print("2025 only — level2_ops data for 2024/2026 not available on Mac.")
    print("Need Windows access to extract raw L2 data for other years.")
    print("Prices for 2024/2026 already downloaded (242/119 days each).")

    # Write the 2025 row
    h5 = summary_df[summary_df["holding_days"] == 5].iloc[0]
    yearly = pd.DataFrame([{
        "year": 2025,
        "n_stocks": int(h5["n_stocks"]),
        "n_trades": int(h5["n_trades"]),
        "avg_ret_5d": h5["avg_ret_pct"],
        "win_5d": h5["win_rate"],
        "total_return": h5["total_return_pct"],
        "max_drawdown": h5["max_drawdown_pct"],
        "best_stock": h5["top_stock"],
        "best_stock_contribution": h5["top_stock_pct"],
        "worst_stock": h5["worst_stock"],
        "monthly_positive_rate": h5["monthly_positive_rate"],
        "note": "with tradeability filters",
    }])
    yearly_path = OUT_DIR / "yearly_oos_backtest_summary.csv"
    yearly.to_csv(yearly_path, index=False)
    print(f"Saved: {yearly_path}")
    print(yearly.to_string(index=False))

    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
