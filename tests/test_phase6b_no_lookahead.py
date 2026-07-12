"""Phase 6B 无未来函数 / 无泄漏回归测试。

保证：
  1. 逐折拟合只由该折 train 标签决定——篡改 test 期标签不改变入选因子/方向；
  2. 滚动窗 train 受 train_start 限制且不含 test 期（purge 后 label 结束早于首 test 日）；
  3. train 窗口确实被 train_start 收窄（expanding train 行数 > rolling train 行数）；
  4. 组合构造为 T+1 建仓——fill_slots 每笔 entry_date 严格晚于 signal_date。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.fusion import alpha_rule_fusion as arf
from src.backtest import portfolio_construction as pc

FOLD_TRAIN_END = "2025-08-31"
FOLD_TEST_START = "2025-09-01"
ROLL_START = "2025-05-01"


@pytest.fixture(scope="module")
def data():
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd((5, 10))
    return panel, fwd


# ---------- 1. 逐折拟合忽略 test 期标签 ----------
def test_fold_fit_ignores_test_period_labels(data):
    panel, fwd = data
    fit = arf.fit_fusion(panel, fwd, 5, train_end=FOLD_TRAIN_END, test_start=FOLD_TEST_START)
    fwd2 = fwd.copy()
    m = fwd2["trade_date"] >= pd.Timestamp(FOLD_TEST_START)
    fwd2.loc[m, "fwd_5d"] = -fwd2.loc[m, "fwd_5d"]
    fwd2.loc[m, "fwd_10d"] = fwd2.loc[m, "fwd_10d"] * 2 - 5
    fit2 = arf.fit_fusion(panel, fwd2, 5, train_end=FOLD_TRAIN_END, test_start=FOLD_TEST_START)
    assert fit["kept"] == fit2["kept"]
    for name in fit["schemes"]:
        assert fit["schemes"][name]["signs"] == fit2["schemes"][name]["signs"], name
        for f, w in fit["schemes"][name]["weights"].items():
            assert abs(w - fit2["schemes"][name]["weights"][f]) < 1e-12


# ---------- 2. 滚动窗 train 受 train_start 限制且不含 test ----------
def test_rolling_fold_train_excludes_test(data):
    panel, _ = data
    sub = panel[panel["trade_date"] >= pd.Timestamp(ROLL_START)]
    train_panel, test_panel, info = arf.purge_train_test(
        sub, 5, train_end=FOLD_TRAIN_END, test_start=FOLD_TEST_START)
    assert train_panel["trade_date"].min() >= pd.Timestamp(ROLL_START)
    assert train_panel["trade_date"].max() < pd.Timestamp(FOLD_TEST_START)
    assert test_panel["trade_date"].min() >= pd.Timestamp(FOLD_TEST_START)
    assert (pd.Timestamp(info["last_train_label_end_date"])
            < pd.Timestamp(info["first_test_trade_date"]))


# ---------- 3. train 窗口确实被 train_start 收窄 ----------
def test_train_window_restricted_by_start(data):
    panel, _ = data
    exp_train, _, _ = arf.purge_train_test(panel, 5, train_end=FOLD_TRAIN_END, test_start=FOLD_TEST_START)
    sub = panel[panel["trade_date"] >= pd.Timestamp(ROLL_START)]
    roll_train, _, _ = arf.purge_train_test(sub, 5, train_end=FOLD_TRAIN_END, test_start=FOLD_TEST_START)
    # 同 train_end 下，expanding（全起点）train 行数应严格多于 rolling（2025-05 起）
    assert len(exp_train) > len(roll_train) > 0
    assert roll_train["trade_date"].min() > exp_train["trade_date"].min()


# ---------- 4. 组合构造 T+1 建仓（fill_slots）----------
def test_fill_slots_entry_strictly_after_signal():
    dates = ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05", "2025-09-08"]
    prices = {}
    for s, g in {"000001": 0.02, "000002": 0.0}.items():
        opens = [10.0 * (1 + g) ** i for i in range(len(dates))]
        prices[s] = pd.DataFrame({"date_str": dates, "open_yuan": opens,
                                  "close_yuan": [o * 1.001 for o in opens]})
    rows = []
    for d in dates:
        for r, s in enumerate(("000001", "000002")):
            rows.append({"trade_date": pd.Timestamp(d), "symbol": s, "final_score": float(2 - r)})
    scores = pd.DataFrame(rows)
    cfg = pc.PortfolioConfig(top_n=1, holding_days=2, cost_bps=20, slippage_bps=10)
    res = pc.run_fill_slots(scores, prices, None, cfg)
    tr = res["trades"]
    assert not tr.empty
    for _, t in tr.iterrows():
        assert t["entry_date"] > t["signal_date"]        # T+1 建仓，无 T+0
