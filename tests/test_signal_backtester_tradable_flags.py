"""Phase 4.6 tradable-flags tests (toy: 1 stock × 10 days).

验证:
  - 涨停(buyable_flag=False) → T+1 不建仓。
  - tradable_flag=False（即使 buyable=True）→ 不建仓（买入须 buyable 且 tradable）。
  - 停牌(suspend_flag=True) 入场日 → 不建仓。
  - 跌停(sellable_flag=False) 到期日 → 不丢失退出事件：pending_exit，下一可卖日按开盘卖出(deferred)。
  - 停牌到期日 → 顺延到下一可交易日卖出，退出事件不丢失。
  - 无 flags 时正常建仓（基线）。

可直接运行: python3 tests/test_signal_backtester_tradable_flags.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig

DATES = [f"2025-01-{d:02d}" for d in range(1, 11)]
STOCK = "000001"


def make_prices(rows: list) -> dict:
    df = pd.DataFrame(rows, columns=["o", "h", "l", "c"]).astype(float)
    df.insert(0, "date_str", DATES[: len(rows)])
    for f, src in [("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")]:
        df[f"{f}_yuan"] = df[src]
    return {STOCK: df.reset_index(drop=True)}


def flat(px=10.0):
    return make_prices([(px, px, px, px)] * len(DATES))


def sig(date):
    return pd.DataFrame([{"stock_code": STOCK, "signal_date": date}])


def flags(rows: list) -> pd.DataFrame:
    """rows: [(date, buyable, sellable, suspend, tradable)]。缺省的 stock-day 视为可交易。"""
    return pd.DataFrame(rows, columns=[
        "trade_date", "buyable_flag", "sellable_flag", "suspend_flag", "tradable_flag"
    ]).assign(symbol=STOCK)


def cfg(**kw):
    base = dict(holding_days=1, max_positions=1, initial_capital=1.0,
                cost_bps=0, slippage_bps=0, name="flags")
    base.update(kw)
    return BacktestConfig(**base)


# ---------------------------------------------------------------------------

def test_baseline_no_flags_enters():
    """无 flags：正常建仓（基线）。"""
    res = SignalBacktester(cfg()).run(sig("2025-01-03"), flat(), tradable_flags=None)
    assert len(res["trades"]) == 1


def test_limit_up_not_buyable():
    """T+1(d3) 涨停 buyable=False → 不建仓。"""
    f = flags([("2025-01-04", False, True, False, True)])  # d3 涨停
    res = SignalBacktester(cfg()).run(sig("2025-01-03"), flat(), tradable_flags=f)
    assert len(res["trades"]) == 0, "涨停应无法买入"
    assert res["equity_curve"]["n_positions"].max() == 0


def test_not_tradable_blocks_buy_even_if_buyable():
    """buyable=True 但 tradable=False → 仍不建仓（买入须同时满足）。"""
    f = flags([("2025-01-04", True, True, False, False)])  # d3 tradable=False
    res = SignalBacktester(cfg()).run(sig("2025-01-03"), flat(), tradable_flags=f)
    assert len(res["trades"]) == 0, "tradable_flag=False 应阻止买入"


def test_suspend_on_entry_day_blocks_buy():
    """入场日 d3 停牌 → 不建仓。"""
    f = flags([("2025-01-04", True, True, True, True)])  # d3 停牌
    res = SignalBacktester(cfg()).run(sig("2025-01-03"), flat(), tradable_flags=f)
    assert len(res["trades"]) == 0


def test_limit_down_defers_exit_to_next_sellable_open():
    """到期日 d4 跌停不可卖 → pending_exit；d5 可卖，按 d5 开盘卖出(deferred)。"""
    # d3 entry open=10；d4 maturity 但跌停；d5 open=9（用于校验成交价）
    rows = [(10, 10, 10, 10)] * 3          # d0..d2
    rows += [(10, 10, 10, 10)]             # d3 entry
    rows += [(10, 10, 10, 10)]             # d4 maturity（跌停，不可卖）
    rows += [(9, 9, 9, 9)]                 # d5 可卖，开盘=9
    rows += [(9, 9, 9, 9)] * 4             # d6..d9
    prices = make_prices(rows)
    f = flags([("2025-01-05", True, False, False, True)])  # d4 跌停 sellable=False
    res = SignalBacktester(cfg()).run(sig("2025-01-03"), prices, tradable_flags=f)
    t = res["trades"]
    assert len(t) == 1, f"expected 1 trade, got {len(t)}"
    row = t.iloc[0]
    assert row["exit_date"] == "2025-01-06", f"deferred exit date={row['exit_date']}"
    assert row["exit_reason"] == "maturity"
    assert bool(row["deferred"]) is True, "应标记 deferred"
    # 成交价 = d5 开盘 9（无滑点）
    assert abs(row["exit_price"] - 9.0) < 1e-9, row["exit_price"]


def test_suspend_on_exit_day_does_not_lose_exit():
    """到期日 d4 停牌 → 顺延到下一可交易日 d5 卖出，退出事件不丢失。"""
    rows = [(10, 10, 10, 10)] * 3
    rows += [(10, 10, 10, 10)]             # d3 entry
    rows += [(10, 10, 10, 10)]             # d4 maturity（停牌）
    rows += [(10, 10, 10, 10)]             # d5 可交易
    rows += [(10, 10, 10, 10)] * 4
    prices = make_prices(rows)
    f = flags([("2025-01-05", True, True, True, True)])  # d4 停牌
    res = SignalBacktester(cfg()).run(sig("2025-01-03"), prices, tradable_flags=f)
    t = res["trades"]
    assert len(t) == 1, "退出事件不应丢失"
    assert t.iloc[0]["exit_date"] == "2025-01-06", t.iloc[0]["exit_date"]
    assert t.iloc[0]["exit_reason"] == "maturity"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tradable-flags tests passed.")


if __name__ == "__main__":
    _run_all()
