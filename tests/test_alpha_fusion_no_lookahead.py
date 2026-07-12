"""Phase 6A 无未来函数 / 无泄漏回归测试。

保证：
  1. 逐 horizon purge 后，train 尾部信号日的 label 结束日 **严格早于** 首 test 日；
  2. 因子筛选/方向/权重只由 train 标签决定——篡改 test 期标签不改变拟合结果；
  3. 打分是纯截面构造——某日打分只依赖当日截面，与其他日数据无关（无跨日泄漏）。
"""
import numpy as np
import pandas as pd
import pytest

from src.fusion import alpha_rule_fusion as arf


@pytest.fixture(scope="module")
def ctx():
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd((5, 10))
    fit5 = arf.fit_fusion(panel, fwd, 5)
    return panel, fwd, fit5


def test_purge_label_end_strictly_before_test_start(ctx):
    panel, _, _ = ctx
    for h in (5, 10):
        train_panel, test_panel, info = arf.purge_train_test(panel, h)
        assert (pd.Timestamp(info["last_train_label_end_date"])
                < pd.Timestamp(info["first_test_trade_date"]))
        # train 面板不含 test_start 及之后的信号日
        assert train_panel["trade_date"].max() < pd.Timestamp(arf.TEST_START)
        assert test_panel["trade_date"].min() >= pd.Timestamp(arf.TEST_START)


def test_fit_ignores_test_period_labels(ctx):
    """篡改 test 期标签后重新 fit，入选因子/方向/权重必须完全不变（train-only 选参）。"""
    panel, fwd, fit5 = ctx
    fwd2 = fwd.copy()
    mask = fwd2["trade_date"] >= pd.Timestamp(arf.TEST_START)
    fwd2.loc[mask, "fwd_5d"] = -fwd2.loc[mask, "fwd_5d"]        # 反号
    fwd2.loc[mask, "fwd_10d"] = fwd2.loc[mask, "fwd_10d"] * 3 + 1  # 任意扰动
    fit2 = arf.fit_fusion(panel, fwd2, 5)

    assert fit5["kept"] == fit2["kept"]
    for name in fit5["schemes"]:
        s1, s2 = fit5["schemes"][name], fit2["schemes"][name]
        assert s1["factors"] == s2["factors"], name
        assert s1["signs"] == s2["signs"], name
        for f in s1["weights"]:
            assert abs(s1["weights"][f] - s2["weights"][f]) < 1e-12, (name, f)


def test_scores_are_cross_sectionally_local(ctx):
    """某日打分只用当日截面：整段面板 vs 只喂当日单日切片，结果对该日逐票一致。"""
    panel, fwd, fit5 = ctx
    _, test_panel, _ = arf.purge_train_test(panel, 5)
    scheme = fit5["schemes"]["equal_weight"]
    full = arf.build_scheme_scores(test_panel, scheme)

    a_date = sorted(test_panel["trade_date"].unique())[10]
    one = arf.build_scheme_scores(test_panel[test_panel["trade_date"] == a_date], scheme)

    merged = (full[full["trade_date"] == a_date]
              .merge(one, on=["trade_date", "symbol"], suffixes=("_full", "_one")))
    assert len(merged) > 0
    diff = (merged["final_score_full"] - merged["final_score_one"]).abs().max()
    assert diff < 1e-12
