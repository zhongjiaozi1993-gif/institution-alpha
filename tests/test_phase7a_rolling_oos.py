"""Phase 7A 无未来函数 / 逐 horizon purge 回归测试。

保证：
  1. fit_fusion 对 horizon=20/40/60 正常返回 kept/schemes；
  2. 长 horizon purge 更激进（purged_rows 随 horizon 增加而单调不降）；
  3. 逐折拟合只由该折 train 标签决定——篡改 test 期标签不改变入选因子/方向；
  4. train 窗口受 train_start 限制且不含 test 期（purge 后 label 结束早于首 test 日）；
  5. load_fwd 正确加载 40d/60d 列。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.fusion import alpha_rule_fusion as arf

FOLD_TRAIN_END = "2025-08-31"
FOLD_TEST_START = "2025-09-01"
ROLL_START = "2025-05-01"
LONG_HORIZONS = (20, 40, 60)


@pytest.fixture(scope="module")
def data():
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd(LONG_HORIZONS)
    return panel, fwd


# ---------- 1. fit_fusion 对 20/40/60d 正常返回 ----------
def test_fit_fusion_works_for_long_horizons(data):
    panel, fwd = data
    for h in LONG_HORIZONS:
        fit = arf.fit_fusion(panel, fwd, h, train_end=FOLD_TRAIN_END,
                             test_start=FOLD_TEST_START, embargo=arf.EMBARGO)
        assert "error" not in fit, f"h={h}d fit_fusion 报错: {fit.get('error')}"
        assert len(fit["kept"]) > 0, f"h={h}d 无因子通过筛选"
        assert "equal_weight" in fit["schemes"], f"h={h}d 缺少 equal_weight"
        assert "best_single" in fit["schemes"], f"h={h}d 缺少 best_single"
        for c in fit["kept"]:
            assert fit["schemes"]["equal_weight"]["signs"][c] in (-1.0, 1.0)


# ---------- 2. 长 horizon purge 更激进 ----------
def test_longer_horizon_purges_more(data):
    panel, fwd = data
    prev = -1
    for h in LONG_HORIZONS:
        fit = arf.fit_fusion(panel, fwd, h, train_end=FOLD_TRAIN_END,
                             test_start=FOLD_TEST_START, embargo=arf.EMBARGO)
        purged = fit["purge"].get("purged_rows", 0)
        # purged_rows 随 horizon 单调不降（更长 horizon → 更多信号日 label 越界）
        assert purged >= prev, f"h={h}d purged={purged} < prev={prev}"
        prev = purged


# ---------- 3. 逐折拟合忽略 test 期标签（20d）----------
def test_fold_fit_ignores_test_period_labels_20d(data):
    panel, fwd = data
    fit = arf.fit_fusion(panel, fwd, 20, train_end=FOLD_TRAIN_END,
                         test_start=FOLD_TEST_START)
    fwd2 = fwd.copy()
    m = fwd2["trade_date"] >= pd.Timestamp(FOLD_TEST_START)
    fwd2.loc[m, "fwd_20d"] = -fwd2.loc[m, "fwd_20d"]
    fit2 = arf.fit_fusion(panel, fwd2, 20, train_end=FOLD_TRAIN_END,
                          test_start=FOLD_TEST_START)
    assert fit["kept"] == fit2["kept"], f"20d kept 受 test 标签影响"
    for name in fit["schemes"]:
        assert fit["schemes"][name]["signs"] == fit2["schemes"][name]["signs"], name
        for f, w in fit["schemes"][name]["weights"].items():
            assert abs(w - fit2["schemes"][name]["weights"][f]) < 1e-12


# ---------- 4. 滚动窗 train 不含 test（20d/40d/60d）----------
def test_rolling_fold_train_excludes_test_all_horizons(data):
    panel, _ = data
    for h in LONG_HORIZONS:
        sub = panel[panel["trade_date"] >= pd.Timestamp(ROLL_START)]
        train_panel, test_panel, info = arf.purge_train_test(
            sub, h, train_end=FOLD_TRAIN_END, test_start=FOLD_TEST_START)
        assert train_panel["trade_date"].min() >= pd.Timestamp(ROLL_START)
        assert train_panel["trade_date"].max() < pd.Timestamp(FOLD_TEST_START)
        assert test_panel["trade_date"].min() >= pd.Timestamp(FOLD_TEST_START)
        assert (pd.Timestamp(info["last_train_label_end_date"])
                < pd.Timestamp(info["first_test_trade_date"])), \
            f"h={h}d label 结束日未早于 test 首日"


# ---------- 5. load_fwd 正确加载 40d/60d ----------
def test_load_fwd_has_long_horizons():
    fwd = arf.load_fwd(LONG_HORIZONS)
    for h in LONG_HORIZONS:
        col = f"fwd_{h}d"
        assert col in fwd.columns, f"缺少 {col}"
        non_null = fwd[col].notna().sum()
        assert non_null > 0, f"{col} 全为 NaN"


# ---------- 6. 40d/60d label 与 fwd 口径一致 ----------
def test_fwd_matches_label_40d_60d():
    lab = pd.read_parquet(PROJ / "data" / "processed" / "labels" / "labels.parquet")
    for h in (40, 60):
        lc = f"label_{h}d"
        fc = f"fwd_{h}d"
        assert lc in lab.columns, f"缺少 {lc}"
        # fwd = label * 100
        fwd_from_label = lab[lc] * 100
        fwd_direct = arf.load_fwd((h,))
        merged = lab[["trade_date", "symbol"]].copy()
        merged["fwd_from_label"] = fwd_from_label
        merged = merged.merge(fwd_direct, on=["trade_date", "symbol"], how="inner")
        diff = (merged["fwd_from_label"] - merged[fc]).abs()
        assert diff.max() < 1e-9, f"h={h}d fwd 与 label×100 不一致, max diff={diff.max()}"
