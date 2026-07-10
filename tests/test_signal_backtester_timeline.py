"""Phase 4.6 timeline & portfolio-accounting tests (toy: up to 3 stocks × 10 days).

验证:
  - T 日信号只在 T+1 入场（entry_date == T+1）。
  - 信号日 T 的 NAV 不含未来仓位（positions_value==0, n_positions==0）。
  - 买入成本不重复扣（入场后 NAV = 1 − 单次买手续费）。
  - 买滑点/卖滑点各体现一次。
  - NAV 可手工复算。
  - 引擎内部 stock_code zfill(6)（未补零输入仍可匹配 prices）。
  - summary.total_return 已删除；portfolio_total_return(NAV) 与 trade_return_sum(单笔) 分离。
  - metrics.compute_full_metrics 接受引擎 trades（net_return_pct schema）。

可直接运行: python3 tests/test_signal_backtester_timeline.py
或 pytest:  pytest tests/test_signal_backtester_timeline.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig
from src.backtest.metrics import compute_full_metrics

DATES = [f"2025-01-{d:02d}" for d in range(1, 11)]  # 10 交易日 d0..d9


def make_prices(spec: dict) -> dict:
    """spec: {stock: [(o,h,l,c) per day]} 对齐 DATES。"""
    prices = {}
    for stock, rows in spec.items():
        df = pd.DataFrame(rows, columns=["o", "h", "l", "c"]).astype(float)
        df.insert(0, "date_str", DATES[: len(rows)])
        for f, src in [("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")]:
            df[f"{f}_yuan"] = df[src]
        prices[stock] = df.reset_index(drop=True)
    return prices


def flat_prices(stock="000001", px=10.0):
    return make_prices({stock: [(px, px, px, px)] * len(DATES)})


def sig(stock, date):
    return pd.DataFrame([{"stock_code": stock, "signal_date": date}])


def _equity_by_date(res):
    return res["equity_curve"].set_index("date")


# ---------------------------------------------------------------------------

def test_entry_at_T_plus_1():
    """信号 d2(01-03) → 入场 d3(01-04)，到期 d4(01-05)。"""
    prices = flat_prices()
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=0, slippage_bps=0, name="t1")
    res = SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices)
    trades = res["trades"]
    assert len(trades) == 1, f"expected 1 trade, got {len(trades)}"
    assert trades.iloc[0]["entry_date"] == "2025-01-04", trades.iloc[0]["entry_date"]
    assert trades.iloc[0]["exit_date"] == "2025-01-05", trades.iloc[0]["exit_date"]
    assert trades.iloc[0]["signal_date"] == "2025-01-03"


def test_signal_day_nav_excludes_future_position():
    """信号日 d2 的 NAV 只含现金，不含尚未建立的仓位。"""
    prices = flat_prices()
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=0, slippage_bps=0, name="t2")
    eq = _equity_by_date(SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices))
    # 信号日 d2：无仓位
    assert eq.loc["2025-01-03", "n_positions"] == 0
    assert eq.loc["2025-01-03", "positions_value"] == 0.0
    # 入场日 d3：1 仓位
    assert eq.loc["2025-01-04", "n_positions"] == 1


def test_buy_cost_not_double_counted():
    """cost=1%, 入场后 NAV = 1 − 单次买手续费 = 0.99（若重复扣则为 0.98）。"""
    prices = flat_prices(px=10.0)
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=100, slippage_bps=0, name="t3")
    res = SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices)
    eq = _equity_by_date(res)
    nav_entry = eq.loc["2025-01-04", "nav"]
    assert abs(nav_entry - 0.99) < 1e-9, f"nav_entry={nav_entry} (double-count?)"
    # 到期卖出后 NAV = 0.9801（买+卖各一次 1% 手续费）
    nav_exit = eq.loc["2025-01-05", "nav"]
    assert abs(nav_exit - 0.9801) < 1e-9, f"nav_exit={nav_exit}"
    # 单笔净收益 ≈ -1.99%（往返 2× cost）
    assert abs(res["trades"].iloc[0]["net_return_pct"] - (-1.99)) < 1e-2


def test_slippage_each_side_once():
    """slip=1%: 买价×1.01, 卖价×0.99；平价下往返约 -1.98%。"""
    prices = flat_prices(px=10.0)
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=0, slippage_bps=100, name="t4")
    res = SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices)
    # 手工: exec_buy=10.1, shares=1/10.1; exec_sell=9.9; nav_exit=shares*9.9
    shares = 1.0 / 10.1
    nav_exit_expected = shares * 9.9
    eq = _equity_by_date(res)
    # NAV 存储到 6 位小数，容差取 1e-6
    assert abs(eq.loc["2025-01-05", "nav"] - nav_exit_expected) < 1e-6, eq.loc["2025-01-05", "nav"]


def test_nav_manual_recompute():
    """价格变动下手工复算 NAV：入场 open=10, 到期 close=12，无成本。"""
    rows = [(10, 10, 10, 10)] * 3          # d0,d1,d2 flat
    rows += [(10, 12, 10, 11)]             # d3 entry: open=10, close=11
    rows += [(11, 13, 11, 12)]             # d4 maturity: close=12
    rows += [(12, 12, 12, 12)] * 5         # d5..d9
    prices = make_prices({"000001": rows})
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=0, slippage_bps=0, name="t5")
    res = SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices)
    eq = _equity_by_date(res)
    # 入场 d3: shares=1/10=0.1; nav_d3=0.1*close(11)=1.1
    assert abs(eq.loc["2025-01-04", "nav"] - 1.1) < 1e-9, eq.loc["2025-01-04", "nav"]
    # 到期 d4: 卖出 close=12 → cash=0.1*12=1.2; nav=1.2
    assert abs(eq.loc["2025-01-05", "nav"] - 1.2) < 1e-9, eq.loc["2025-01-05", "nav"]
    assert abs(res["trades"].iloc[0]["net_return_pct"] - 20.0) < 1e-6


def test_stock_code_zfill_inside_engine():
    """signal stock_code 传入未补零形式（int 1）→ 引擎内部 zfill(6) 匹配 prices('000001')。"""
    prices = flat_prices()  # keyed by "000001"
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=0, slippage_bps=0, name="zf")
    signals = pd.DataFrame([{"stock_code": 1, "signal_date": "2025-01-03"}])  # 未补零 int
    res = SignalBacktester(cfg).run(signals, prices)
    assert len(res["trades"]) == 1, "引擎内部 zfill 后应能匹配 prices 建仓"
    assert res["trades"].iloc[0]["stock"] == "000001", res["trades"].iloc[0]["stock"]


def test_summary_total_return_is_portfolio_not_trade_sum():
    """summary 删除误导性 total_return；portfolio_total_return(NAV) 与 trade_return_sum(单笔求和) 分离。"""
    rows = [(10, 10, 10, 10)] * 3          # d0,d1,d2 flat
    rows += [(10, 12, 10, 11)]             # d3 entry: open=10
    rows += [(11, 13, 11, 12)]             # d4 maturity: close=12
    rows += [(12, 12, 12, 12)] * 5         # d5..d9
    prices = make_prices({"000001": rows})
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=0, slippage_bps=0, name="sum")
    res = SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices)
    cols = res["summary"].columns
    assert "total_return" not in cols, "total_return 应删除（不再等于 trade_return_sum）"
    assert "portfolio_total_return" in cols and "trade_return_sum" in cols
    s = res["summary"].iloc[0]
    # NAV 口径 20% → 0.2（小数）；单笔求和 20.0（百分数）。数值不同即证明两口径分离。
    assert abs(s["portfolio_total_return"] - 0.2) < 1e-6, s["portfolio_total_return"]
    assert abs(s["trade_return_sum"] - 20.0) < 1e-3, s["trade_return_sum"]


def test_metrics_accepts_trades_schema():
    """compute_full_metrics 接受引擎 trades（net_return_pct schema），返回组合+交易口径且不崩。"""
    prices = flat_prices(px=10.0)
    cfg = BacktestConfig(holding_days=1, max_positions=1, initial_capital=1.0,
                         cost_bps=100, slippage_bps=0, name="m")
    res = SignalBacktester(cfg).run(sig("000001", "2025-01-03"), prices)
    m = compute_full_metrics(res["equity_curve"], res["trades"])
    assert "portfolio_total_return" in m, "应含组合 NAV 口径"
    assert "trade_return_sum" in m, "应含单笔求和口径"
    assert m["total_trades"] == 1
    assert "win_rate" in m


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} timeline tests passed.")


if __name__ == "__main__":
    _run_all()
