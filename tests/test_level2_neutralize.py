"""Phase 5.2C 中性化核心逻辑的 sanity 测试（结论依赖其正确性）。"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_l2_attr", PROJ / "scripts" / "run_level2_attribution.py")
attr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attr)


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2025-09-01", "2025-10-31"):
        n = 60
        la = rng.normal(18, 1, n)          # log_amount
        to = rng.uniform(0.005, 0.06, n)   # turnover
        mc = rng.normal(23, 1, n)          # log_mktcap
        feat = 2.0 * la + 0.5 * mc + rng.normal(0, 0.01, n)   # 纯规模线性组合
        for i in range(n):
            rows.append({"trade_date": d, "symbol": f"{i:06d}", "feat": feat[i],
                         "log_amount": la[i], "turnover": to[i], "log_mktcap": mc[i]})
    return pd.DataFrame(rows)


def test_residual_orthogonal_to_controls():
    df = _panel()
    res = attr.neutralize(df, "feat", attr.CTRL)
    m = res.merge(df, on=["trade_date", "symbol"])
    for _, g in m.groupby("trade_date"):
        for c in attr.CTRL:
            assert abs(np.corrcoef(g["resid"], g[c])[0, 1]) < 1e-6


def test_pure_size_feature_neutralizes_to_near_zero():
    df = _panel()
    res = attr.neutralize(df, "feat", attr.CTRL)
    m = res.merge(df, on=["trade_date", "symbol"])
    # 纯规模线性组合 → 残差只剩噪声，方差远小于原始
    assert m["resid"].std() < 0.05 * m["feat"].std()
