"""Phase 6A 规则融合核心逻辑 sanity 测试（结论依赖其正确性）。

覆盖：权重归一/上限、方向对齐、去相关代表选择、单因子方向判定、四方案权重和=1。
用小型合成面板直接调 arf 函数，避免依赖全量数据。
"""
import numpy as np
import pandas as pd

from src.fusion import alpha_rule_fusion as arf


def _one_day_panel(cols_values: dict, date="2025-06-02"):
    """构造单日面板：cols_values = {factor: array}。"""
    n = len(next(iter(cols_values.values())))
    df = pd.DataFrame({"trade_date": pd.Timestamp(date),
                       "symbol": [f"{i:06d}" for i in range(n)]})
    for c, v in cols_values.items():
        df[c] = v
    return df


# ---------- _cap_normalize ----------
def test_cap_normalize_respects_cap_and_sums_to_one():
    # 4 因子 × cap0.30 = 1.2 ≥ 1 可行 → 上限 0.30 生效
    w = arf._cap_normalize({"a": 0.6, "b": 0.25, "c": 0.10, "d": 0.05}, cap=0.30)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.30 + 1e-9
    assert w["a"] >= w["b"] >= w["c"] >= w["d"]   # 排序保持


def test_cap_normalize_single_factor_is_one():
    w = arf._cap_normalize({"a": 5.0}, cap=0.30)
    assert abs(w["a"] - 1.0) < 1e-9           # 单因子上限无意义，必须=1


def test_cap_normalize_infeasible_cap_relaxes():
    # 2 因子 × cap0.3 = 0.6 < 1 不可行 → eff_cap=max(0.3,0.5)=0.5，和仍=1
    w = arf._cap_normalize({"a": 0.9, "b": 0.1}, cap=0.30)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.5 + 1e-9


# ---------- build_scheme_scores 方向对齐 ----------
def test_scheme_score_applies_direction_sign():
    x = np.arange(30, dtype=float)
    panel = _one_day_panel({"x": x})
    scheme = {"factors": ["x"], "signs": {"x": -1.0}, "weights": {"x": 1.0}}
    sc = arf.build_scheme_scores(panel, scheme)
    m = sc.merge(panel, on=["trade_date", "symbol"])
    # sign=-1 → final_score 与原始因子秩负相关
    assert m["final_score"].corr(m["x"], method="spearman") < -0.99


def test_scheme_score_handles_missing_factor_by_reweighting():
    x = np.arange(30, dtype=float)
    y = np.arange(30, dtype=float)[::-1].copy()
    y[:10] = np.nan                            # 前 10 只缺 y
    panel = _one_day_panel({"x": x, "y": y})
    scheme = {"factors": ["x", "y"], "signs": {"x": 1.0, "y": 1.0},
              "weights": {"x": 0.5, "y": 0.5}}
    sc = arf.build_scheme_scores(panel, scheme)
    # 缺 y 的股票只用 x（按可得权重归一）→ final_score 全部有限，无 NaN
    assert sc["final_score"].notna().all()
    assert np.isfinite(sc["final_score"]).all()


# ---------- correlation_prune 去相关 ----------
def test_correlation_prune_drops_duplicate_factor():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.bdate_range("2025-05-01", "2025-06-30"):
        n = 40
        a = rng.normal(0, 1, n)
        rows.append(pd.DataFrame({
            "trade_date": d, "symbol": [f"{i:06d}" for i in range(n)],
            "f1": a, "f2": a.copy(), "f3": rng.normal(0, 1, n)}))  # f2≡f1, f3独立
    panel = pd.concat(rows, ignore_index=True)
    screen_df = pd.DataFrame({
        "factor": ["f1", "f2", "f3"], "sign": [1.0, 1.0, 1.0],
        "pass": [True, True, True], "screen_score": [3.0, 2.0, 1.0]})
    kept, corr = arf.correlation_prune(panel, screen_df, thresh=0.70)
    assert "f1" in kept and "f2" not in kept   # 完全相同 → 只留 screen_score 高者 f1
    assert "f3" in kept                        # 独立 → 保留
    assert abs(corr.loc["f1", "f2"]) > 0.99


# ---------- screen_factor 方向 ----------
def test_screen_factor_sign_matches_rankic():
    rng = np.random.default_rng(1)
    prows, frows = [], []
    for d in pd.bdate_range("2025-05-01", "2025-08-01"):
        n = 50
        f = rng.normal(0, 1, n)
        fwd5 = -2.0 * f + rng.normal(0, 0.5, n)   # 负相关 → sign 应为 -1
        syms = [f"{i:06d}" for i in range(n)]
        prows.append(pd.DataFrame({"trade_date": d, "symbol": syms, "g": f}))
        frows.append(pd.DataFrame({"trade_date": d, "symbol": syms,
                                   "fwd_5d": fwd5, "fwd_10d": fwd5}))
    panel = pd.concat(prows, ignore_index=True)
    fwd = pd.concat(frows, ignore_index=True)
    r = arf.screen_factor(panel, "g", fwd, horizon=5)
    assert r["sign"] == -1.0
    assert r["rankic"] < 0


# ---------- 四方案权重和=1 ----------
def test_weight_schemes_sum_to_one():
    kept = ["a", "b", "c"]
    screen_df = pd.DataFrame({
        "factor": ["a", "b", "c"], "sign": [1.0, -1.0, 1.0],
        "rankicir": [-0.6, 0.4, 0.3], "monthly_consistency": [0.9, 0.8, 0.7],
        "coverage": [0.95, 0.9, 0.85]})
    corr = pd.DataFrame(np.eye(3), index=kept, columns=kept)
    schemes = arf.build_weight_schemes(screen_df, kept, corr)
    for name, sc in schemes.items():
        assert abs(sum(sc["weights"].values()) - 1.0) < 1e-9, name
        if name in ("icir_weight", "stability_weight"):
            # 3 因子时 cap0.3 不可行 → eff_cap=max(cap,1/n)
            assert max(sc["weights"].values()) <= max(arf.WEIGHT_CAP, 1.0 / len(kept)) + 1e-9
    assert schemes["best_single"]["factors"] == ["a"]   # screen_df 首行=最优单因子
