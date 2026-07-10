"""标准回测入口：把连续因子转为每日 Top-N 买入，经统一回测引擎回测。

关键: 接入 data/processed/tradable/tradable_flags.parquet，强制执行
涨停不可买 / 跌停不可卖 / 停牌不可交易（Phase 4.5）。

用法:
    python3 scripts/run_signal_backtest.py \
        --signal-file data/processed/signals/price_alpha191_full/signal027.parquet \
        --universe Universe_B --holding-days 5 --top-n 30 --direction original

    # 关闭交易约束（对照旧口径）:
    python3 scripts/run_signal_backtest.py ... --no-flags
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig
from src.backtest.metrics import compute_full_metrics
from src.registry import universe_registry as reg

DAILY_DIR = PROJECT / "data" / "daily"
FLAGS_PATH = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"
OUT_DIR = PROJECT / "data" / "processed" / "backtest"
PRICE_SCALE = 100  # 日线为后复权「分」，/100 得元（比例不影响收益，仅影响绝对价）


def load_prices(codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    prices = {}
    for code in codes:
        code = str(code).zfill(6)
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        for f in ["open", "high", "low", "close"]:
            df[f"{f}_yuan"] = df[f] / PRICE_SCALE
        prices[code] = df
    return prices


def signal_to_buys(sig: pd.DataFrame, top_n: int, direction: str) -> pd.DataFrame:
    """每个交易日按 signal_value 选 Top-N（inverse 则选 Bottom-N）作为买入信号。"""
    sig = sig.dropna(subset=["signal_value"]).copy()
    ascending = (direction == "inverse")
    buys = []
    for date, g in sig.groupby("signal_date"):
        picked = g.sort_values("signal_value", ascending=ascending).head(top_n)
        for code in picked["stock_code"]:
            buys.append({"stock_code": str(code).zfill(6), "signal_date": date})
    return pd.DataFrame(buys)


def main():
    ap = argparse.ArgumentParser(description="标准信号回测入口（含交易约束）")
    ap.add_argument("--signal-file",
                    default=str(PROJECT / "data/processed/signals/price_alpha191_full/signal027.parquet"))
    ap.add_argument("--universe", default="Universe_B")
    ap.add_argument("--holding-days", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--direction", choices=["original", "inverse"], default="original")
    ap.add_argument("--stop-loss", type=float, default=None)
    ap.add_argument("--take-profit", type=float, default=None)
    ap.add_argument("--max-positions", type=int, default=30)
    ap.add_argument("--cost-bps", type=float, default=20)
    ap.add_argument("--slippage-bps", type=float, default=10)
    ap.add_argument("--start-date", default="2025-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--no-flags", action="store_true", help="关闭涨跌停/停牌约束（对照）")
    ap.add_argument("--output-prefix", default=None)
    args = ap.parse_args()

    # ---- signal + universe ----
    universe = set(reg.load_universe(args.universe))
    sig = pd.read_parquet(args.signal_file)
    sig["signal_date"] = pd.to_datetime(sig["trade_date"]).dt.strftime("%Y-%m-%d")
    sig["stock_code"] = sig["stock_code"].astype(str).str.zfill(6)
    sig = sig[(sig["trade_date"] >= args.start_date) & (sig["trade_date"] <= args.end_date)]
    sig = sig[sig["stock_code"].isin(universe)]
    sid = sig["signal_id"].iloc[0] if "signal_id" in sig.columns and len(sig) else Path(args.signal_file).stem
    print(f"Signal {sid} on {args.universe}: {sig['stock_code'].nunique()} stocks, {sig['signal_date'].nunique()} dates")

    buys = signal_to_buys(sig, args.top_n, args.direction)
    print(f"Daily Top-{args.top_n} ({args.direction}) → {len(buys)} buy signals")

    # ---- prices + flags ----
    codes = sorted(set(buys["stock_code"]))
    prices = load_prices(codes, args.start_date, args.end_date)
    flags = None if args.no_flags else pd.read_parquet(FLAGS_PATH)
    print(f"Prices loaded: {len(prices)} stocks | tradable_flags: {'OFF' if args.no_flags else 'ON'}")

    # ---- run ----
    cfg = BacktestConfig(
        holding_days=args.holding_days, stop_loss=args.stop_loss, take_profit=args.take_profit,
        max_positions=args.max_positions, cost_bps=args.cost_bps, slippage_bps=args.slippage_bps,
        name=f"{sid}_{args.universe}_{args.direction}_h{args.holding_days}",
    )
    bt = SignalBacktester(cfg)
    result = bt.run(buys, prices, tradable_flags=flags)
    trades, equity, summary = result["trades"], result["equity_curve"], result["summary"]

    # ---- metrics ----
    nav_metrics = compute_full_metrics(equity, pd.DataFrame())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or cfg.name + ("" if not args.no_flags else "_noflags")
    trades.to_csv(OUT_DIR / f"{prefix}_trades.csv", index=False)
    equity.to_csv(OUT_DIR / f"{prefix}_equity.csv", index=False)

    print("\n=== 回测结果 ===")
    print(f"配置: {cfg.name} | flags={'ON' if not args.no_flags else 'OFF'}")
    if not summary.empty:
        s = summary.iloc[0]
        print(f"  交易数: {s['n_trades']}  股票数: {s['n_stocks']}  胜率: {s['win_rate']*100:.1f}%")
        print(f"  平均单笔净收益: {s['avg_ret']:+.3f}%  单笔收益求和: {s['total_return']:+.2f}%(非组合口径)")
        print(f"  最大回撤(NAV): {s['max_drawdown']*100:.2f}%  月度正收益比: {s['monthly_positive_rate']*100:.0f}%")
        print(f"  最大单票贡献: {s['best_stock']} {s['best_stock_contribution']:+.1f}%  "
              f"(占累计 {s['best_stock_contribution']/s['total_return']*100:.0f}%)" if s['total_return'] else "")
        print(f"  止损/止盈退出: {s['stop_loss_exits']}/{s['take_profit_exits']}")
    for k in ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio"]:
        if k in nav_metrics:
            print(f"  {k}: {nav_metrics[k]}")
    print(f"\n输出: {OUT_DIR}/{prefix}_trades.csv, _equity.csv")


if __name__ == "__main__":
    main()
