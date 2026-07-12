"""Level-2 特征验证器（Phase 5 / 5.1）。

复用 Alpha191 验证管线的核心函数（factor_validator._daily_corr /
_quintile_spread_turnover），但改用 **open-to-open 超额收益**（label_*d_excess_index）
作为 fwd，在 **Universe_C**（有 Level-2 数据的股票）上验证每个 Level-2 特征。

产出（暂不做 ML）:
  1. 单特征稳定性: 各 horizon 的 RankIC / RankICIR / spread + IC/spread 各自有效天数。
  2. 与 Alpha191 的正交性（5.1 修正）: **按 trade_date 每日 Spearman** 后汇总
     mean / median / abs_mean / n_days（不再 pooled）。
  3. 样本内增量: 全样本 IC 加权融合 vs 单独。
  4. OOS 增量（5.1 新增）: train(≤8月)选 top-k/方向/权重，test(≥9月)固定参数只评估，
     对比 Alpha191 单独 / L2 综合分 / 融合。

注: 超额 = label − 指数收益（同日对所有股票是同一常数），因此 RankIC / 多空 spread
与用原始 label 完全一致；超额仅令分位绝对收益反映“是否跑赢中证1000”。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation import factor_validator as fv
from src.features.level2_feature_builder import FEATURE_NAMES

PROJECT = Path(__file__).resolve().parent.parent.parent
LABELS = PROJECT / "data" / "processed" / "labels" / "labels.parquet"
L2_FEATURES = PROJECT / "data" / "processed" / "level2" / "level2_daily_features.parquet"
ALPHA_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191_full"

HORIZONS = fv.HORIZONS  # [1,3,5,10]
COST = fv.COST_ROUNDTRIP_PCT


def load_excess_fwd() -> pd.DataFrame:
    """label 表 → fwd_{h}d（open-to-open 超额, 百分比）。"""
    lab = pd.read_parquet(LABELS)
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["symbol"] = lab["symbol"].astype(str).str.zfill(6)
    out = lab[["trade_date", "symbol"]].copy()
    for h in HORIZONS:
        out[f"fwd_{h}d"] = lab[f"label_{h}d_excess_index"] * 100
    return out


def load_l2_features() -> pd.DataFrame:
    df = pd.read_parquet(L2_FEATURES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df


def validate_feature(feat_df: pd.DataFrame, fwd: pd.DataFrame, feature: str) -> dict:
    """单个 Level-2 特征在其可得的 stock-day 上的截面稳定性。"""
    sig = feat_df[["trade_date", "symbol", feature]].rename(columns={feature: "signal_value"})
    merged = sig.merge(fwd, on=["trade_date", "symbol"], how="left")
    res = {"feature": feature,
           "n_obs": len(sig), "n_stocks": sig["symbol"].nunique(),
           "n_dates": sig["trade_date"].nunique(),
           "nonzero_pct": float((sig["signal_value"] != 0).mean()) * 100}
    for h in HORIZONS:
        fc = f"fwd_{h}d"
        res[f"coverage_{h}d"] = merged[fc].notna().mean() * 100
        ic = fv._daily_corr(merged, fc)
        if len(ic):
            std = ic["RankIC"].std()
            res[f"RankIC_{h}d"] = float(ic["RankIC"].mean())
            res[f"RankICIR_{h}d"] = float(ic["RankIC"].mean() / std) if std > 0 else 0.0
            res[f"ic_n_days_{h}d"] = int(len(ic))
        else:
            res[f"RankIC_{h}d"] = res[f"RankICIR_{h}d"] = np.nan
            res[f"ic_n_days_{h}d"] = 0
        spread, topq, botq, top_sets = fv._quintile_spread_turnover(merged, fc)
        res[f"spread_{h}d"] = spread
        res[f"cost_adj_spread_{h}d"] = spread - COST if not np.isnan(spread) else np.nan
        res[f"spread_n_days_{h}d"] = len(top_sets)   # spread 实际有效天数（≠ IC 有效日）
    return res


def load_alpha_on_l2_grid(feat_df: pd.DataFrame) -> pd.DataFrame:
    """把 30 个 Alpha191 信号裁到 Level-2 的 (trade_date, symbol) 网格，宽表返回。"""
    grid = feat_df[["trade_date", "symbol"]].drop_duplicates()
    wide = grid.copy()
    for fp in sorted(ALPHA_DIR.glob("signal*.parquet")):
        s = pd.read_parquet(fp)
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        sid = str(s["signal_id"].iloc[0]) if "signal_id" in s.columns else fp.stem
        col = s[["trade_date", "stock_code", "signal_value"]].rename(
            columns={"stock_code": "symbol", "signal_value": f"a_{sid.lower()}"})
        col["symbol"] = col["symbol"].astype(str).str.zfill(6)
        wide = wide.merge(col, on=["trade_date", "symbol"], how="left")
    return wide


def _rankic_of_column(merged: pd.DataFrame, col: str, fwd_col: str) -> tuple[float, float, int]:
    """给定已并入 fwd 的表，返回某列 (RankIC均值, RankICIR, 有效天数)。"""
    tmp = merged[["trade_date", col, fwd_col]].rename(columns={col: "signal_value"})
    ic = fv._daily_corr(tmp, fwd_col)
    if not len(ic):
        return np.nan, np.nan, 0
    std = ic["RankIC"].std()
    return float(ic["RankIC"].mean()), (float(ic["RankIC"].mean() / std) if std > 0 else 0.0), int(len(ic))


def _eval_signal(merged: pd.DataFrame, col: str, fwd_col: str) -> dict:
    """给定并入 fwd 的表，评估某信号列: RankIC/RankICIR/IC有效日 + spread/扣费spread/spread有效日。"""
    tmp = merged[["trade_date", "symbol", col, fwd_col]].rename(columns={col: "signal_value"})
    ic = fv._daily_corr(tmp, fwd_col)
    if len(ic):
        std = ic["RankIC"].std()
        rankic = float(ic["RankIC"].mean())
        rankicir = float(rankic / std) if std > 0 else 0.0
    else:
        rankic = rankicir = np.nan
    spread, topq, botq, top_sets = fv._quintile_spread_turnover(tmp, fwd_col)
    return {
        "rankic": rankic, "rankicir": rankicir, "ic_n_days": int(len(ic)),
        "spread": spread, "cost_adj_spread": (spread - COST if not np.isnan(spread) else np.nan),
        "spread_n_days": len(top_sets),
    }


def orthogonality(feat_df: pd.DataFrame, alpha_wide: pd.DataFrame,
                  l2_feature: str, top_k: int = 8, min_per_day: int = 5,
                  min_days: int = 5) -> pd.DataFrame:
    """Level-2 特征与各 Alpha191 的**每日** Spearman 相关，再汇总（不再 pooled）。

    每个交易日截面上算 Spearman（≥min_per_day 只），再对各日相关汇总:
      mean / median / abs_mean / n_days。返回按 abs_mean 降序的 top_k。
    """
    m = feat_df[["trade_date", "symbol", l2_feature]].merge(
        alpha_wide, on=["trade_date", "symbol"], how="inner")
    a_cols = [c for c in alpha_wide.columns if c.startswith("a_")]
    rows = []
    for a in a_cols:
        daily = []
        for _, g in m.groupby("trade_date"):
            v = g[[l2_feature, a]].dropna()
            if len(v) < min_per_day:
                continue
            c = v[l2_feature].corr(v[a], method="spearman")
            if pd.notna(c):
                daily.append(c)
        if len(daily) < min_days:
            continue
        arr = np.asarray(daily, dtype=float)
        rows.append({"alpha": a.replace("a_", ""),
                     "mean": round(float(arr.mean()), 4),
                     "median": round(float(np.median(arr)), 4),
                     "abs_mean": round(float(np.abs(arr).mean()), 4),
                     "n_days": int(len(arr))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_mean", ascending=False).head(top_k).reset_index(drop=True)


def _select_composite_params(feat_df: pd.DataFrame, fwd: pd.DataFrame, features: list[str],
                             horizon: int = 5, k: int = 6) -> tuple[list[str], dict]:
    """在给定样本上按 |RankICIR_{h}d| 选 top-k 特征并定方向（符号=RankIC 符号）。

    仅用传入的 feat_df（OOS 时传 train 子集），不看 test。返回 (入选列, 符号)。
    """
    scored = []
    for c in features:
        r = validate_feature(feat_df, fwd, c)
        scored.append((c, r.get(f"RankIC_{horizon}d", np.nan), r.get(f"RankICIR_{horizon}d", np.nan)))
    scored = [s for s in scored if not np.isnan(s[1])]
    scored.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else 0, reverse=True)
    chosen = scored[:k]
    signs = {c: (1.0 if ic >= 0 else -1.0) for c, ic, _ in chosen}
    return [c for c, _, _ in chosen], signs


def _apply_composite(feat_df: pd.DataFrame, cols: list[str], signs: dict) -> pd.DataFrame:
    """把已定的特征/方向应用到给定 stock-day：逐日 rank→符号→z 分→等权，得 l2_composite。"""
    parts = []
    for date, g in feat_df.groupby("trade_date"):
        if len(g) < 5:
            continue
        g = g.copy()
        z_sum = np.zeros(len(g))
        for c in cols:
            v = g[c].rank() * signs[c]
            z = (v - v.mean()) / v.std() if v.std() > 0 else v * 0
            z_sum = z_sum + z.to_numpy()
        g["l2_composite"] = z_sum / max(len(cols), 1)
        parts.append(g[["trade_date", "symbol", "l2_composite"]])
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["trade_date", "symbol", "l2_composite"]))


def build_l2_composite(feat_df: pd.DataFrame, fwd: pd.DataFrame, features: list[str],
                       horizon: int = 5, k: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """全样本综合分：top-k（按 |RankICIR_{h}d|）符号对齐等权 z 分。返回 (含 l2_composite 表, 入选列)。

    注: 全样本选择/对齐 → **样本内**方向性证据；严格 OOS 见 oos_validation。
    """
    cols, signs = _select_composite_params(feat_df, fwd, features, horizon, k)
    return _apply_composite(feat_df, cols, signs), cols


def incremental_test(alpha_wide: pd.DataFrame, fwd: pd.DataFrame,
                     l2_composite: pd.DataFrame, best_alpha_col: str, horizon: int = 5) -> dict:
    """无 ML 增量证明：IC 加权融合 (Alpha191 最优 + Level-2 综合分) vs 单独，比较 RankIC_{h}d。

    融合 = w_a·z(rank(alpha)·sign_a) + w_l·z(rank(l2_comp))，权重 w=各自 |RankIC|。
    弱但正交的 Level-2 综合分在 IC 加权下应抬升组合 RankIC。
    """
    fc = f"fwd_{horizon}d"
    base = (l2_composite.merge(alpha_wide[["trade_date", "symbol", best_alpha_col]],
                               on=["trade_date", "symbol"], how="inner")
            .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left"))
    base = base.dropna(subset=["l2_composite", best_alpha_col, fc])
    if base.empty:
        return {"error": "融合样本不足", "horizon": horizon}

    ic_a, icir_a, _ = _rankic_of_column(base, best_alpha_col, fc)
    ic_l, icir_l, _ = _rankic_of_column(base, "l2_composite", fc)
    sign_a = np.sign(ic_a) if not np.isnan(ic_a) and ic_a != 0 else 1.0
    sign_l = np.sign(ic_l) if not np.isnan(ic_l) and ic_l != 0 else 1.0
    w_a = abs(ic_a) if not np.isnan(ic_a) else 0.0
    w_l = abs(ic_l) if not np.isnan(ic_l) else 0.0
    if w_a + w_l == 0:
        return {"error": "IC 全为 0", "horizon": horizon}

    parts = []
    for date, g in base.groupby("trade_date"):
        if len(g) < 5:
            continue
        g = g.copy()
        ra = g[best_alpha_col].rank() * sign_a
        rl = g["l2_composite"].rank() * sign_l
        za = (ra - ra.mean()) / ra.std() if ra.std() > 0 else ra * 0
        zl = (rl - rl.mean()) / rl.std() if rl.std() > 0 else rl * 0
        g["fused"] = w_a * za + w_l * zl
        parts.append(g[["trade_date", "fused", fc]])
    fused = pd.concat(parts, ignore_index=True)
    ic_f = fv._daily_corr(fused.rename(columns={"fused": "signal_value"}), fc)
    ic_fused = float(ic_f["RankIC"].mean()) if len(ic_f) else np.nan
    icir_fused = (float(ic_f["RankIC"].mean() / ic_f["RankIC"].std())
                  if len(ic_f) and ic_f["RankIC"].std() > 0 else np.nan)

    return {
        "horizon": horizon, "best_alpha": best_alpha_col.replace("a_", ""),
        "n_days": int(len(ic_f)) if len(ic_f) else 0,
        "rankic_alpha": round(ic_a, 4), "rankicir_alpha": round(icir_a, 3),
        "rankic_l2comp": round(ic_l, 4), "rankicir_l2comp": round(icir_l, 3),
        "rankic_fused": round(ic_fused, 4), "rankicir_fused": round(icir_fused, 3),
        "abs_gain_vs_alpha": round(abs(ic_fused) - abs(ic_a), 4),
        "incremental": bool(abs(ic_fused) > abs(ic_a) + 1e-6),
    }


def pick_best_alpha(alpha_wide: pd.DataFrame, fwd: pd.DataFrame, horizon: int = 5) -> tuple[str, float]:
    """在 Level-2 网格上，按 |RankIC_{h}d| 选最强 Alpha191 参照列。"""
    fc = f"fwd_{horizon}d"
    m = alpha_wide.merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left")
    best_col, best_ic = None, 0.0
    for a in [c for c in alpha_wide.columns if c.startswith("a_")]:
        ic, _, nd = _rankic_of_column(m.dropna(subset=[a, fc]), a, fc)
        if not np.isnan(ic) and abs(ic) > abs(best_ic):
            best_col, best_ic = a, ic
    return best_col, best_ic


def _fuse_column(base: pd.DataFrame, alpha_col: str, comp_col: str,
                 w_a: float, w_l: float, sign_a: float, sign_l: float) -> pd.DataFrame:
    """逐日构造融合列: fused = w_a·z(rank(alpha)·sign_a) + w_l·z(rank(comp)·sign_l)。"""
    parts = []
    for _, g in base.groupby("trade_date"):
        if len(g) < 5:
            continue
        g = g.copy()
        ra = g[alpha_col].rank() * sign_a
        rl = g[comp_col].rank() * sign_l
        za = (ra - ra.mean()) / ra.std() if ra.std() > 0 else ra * 0
        zl = (rl - rl.mean()) / rl.std() if rl.std() > 0 else rl * 0
        g["fused"] = w_a * za + w_l * zl
        parts.append(g[["trade_date", "symbol", "fused"]])
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["trade_date", "symbol", "fused"]))


def alpha_direction_flipped(sign_a: float, raw_test_ic: float) -> bool:
    """参照 alpha 方向是否在 OOS 反转：train 原始符号(sign_a) 与 test 原始 RankIC 异号。

    注意：不能比较 train-定向 RankIC 与未定向 raw（当 sign_a=-1 时两者恒异号，会误报反转）。
    正确判据 = sign(train 原始 IC) 与 sign(test 原始 IC) 是否相反，即 sign_a * raw < 0。
    """
    return bool((not np.isnan(raw_test_ic)) and (sign_a * raw_test_ic < 0))


def purge_split_info(all_dates, train_end: str, test_start: str,
                     horizon: int, embargo: int = 6) -> tuple[pd.Timestamp | None, dict]:
    """按 **label_end_date < test_start** 做逐 horizon purge，再叠加 embargo 交易日额外隔离。

    label_hd 用 open[T+1+h]，故信号日 T（交易日历位置 i）的 label 结束日 = 位置 i+1+h。
    - horizon purge：只保留 all_dates[i+1+h] < test_start 的信号日（最大位置 = pos_first_test-(h+2)）；
    - embargo：在此基础上再往前多留 embargo 个交易日空档（额外隔离，不替代 horizon purge）。
    返回 (train_cut_date, info)。info 含 last_train_trade_date / last_train_label_end_date /
    first_test_trade_date / horizon_cut_date，供报告披露。
    """
    ad = np.sort(np.asarray(all_dates, dtype="datetime64[ns]"))
    n = len(ad)
    ts = np.datetime64(pd.Timestamp(test_start))
    te = np.datetime64(pd.Timestamp(train_end))
    pos = int(np.searchsorted(ad, ts))                  # 首个 >= test_start 的位置
    first_test = ad[pos] if pos < n else None
    horizon_cut = pos - (horizon + 2)                   # 最大 i 使 all_dates[i+1+h] < test_start
    cut = horizon_cut - int(embargo)
    te_pos = int(np.searchsorted(ad, te, side="right")) - 1   # 不超过传入 train_end
    if te_pos >= 0:
        cut = min(cut, te_pos)
        horizon_cut = min(horizon_cut, te_pos)

    def _d(p):
        return str(pd.Timestamp(ad[p]).date()) if 0 <= p < n else None

    train_cut_date = pd.Timestamp(ad[cut]) if 0 <= cut < n else None
    info = {
        "horizon": horizon, "embargo": int(embargo),
        "first_test_trade_date": None if first_test is None else str(pd.Timestamp(first_test).date()),
        "last_train_trade_date": _d(cut),
        "last_train_label_end_date": _d(cut + 1 + horizon) if 0 <= cut < n else None,
        "horizon_cut_date": _d(horizon_cut),
    }
    return train_cut_date, info


def oos_validation(feat_df: pd.DataFrame, alpha_wide: pd.DataFrame, fwd: pd.DataFrame,
                   features: list[str], horizon: int = 5, k: int = 6,
                   train_end: str = "2025-08-31", test_start: str = "2025-09-01",
                   embargo: int | None = None) -> dict:
    """样本外增量：train 选参（top-k/方向/权重/最优 alpha），test 固定参数只评估。

    test 上三者在**同一批 stock-day**（综合分∩alpha∩fwd）上比较:
      Alpha191 单独 / L2 综合分 / 融合 → RankIC_{h}d / RankICIR_{h}d / spread_{h}d。
    无 test 端信息用于选择，避免泄漏。

    **purge/embargo**（见 purge_split_info）：label_hd 用 open[T+1+h]，train 尾部信号日的
    label 会越过 test_start 窥探 test 期。先按 **label_end_date < test_start** 逐 horizon 剔除，
    再叠加 embargo（默认 6）交易日额外隔离。
    """
    fc = f"fwd_{horizon}d"
    te, ts = pd.Timestamp(train_end), pd.Timestamp(test_start)
    emb = 6 if embargo is None else int(embargo)

    # purge：label_end_date < test_start（逐 horizon）+ embargo 额外隔离
    train_cut, purge_info = purge_split_info(feat_df["trade_date"].unique(),
                                             train_end, test_start, horizon, emb)
    if train_cut is None:
        return {"error": "purge 后无 train 样本", "horizon": horizon}
    cand = feat_df[feat_df["trade_date"] <= te]
    train_feat = feat_df[feat_df["trade_date"] <= train_cut]
    test_feat = feat_df[feat_df["trade_date"] >= ts]
    hcut = pd.Timestamp(purge_info["horizon_cut_date"]) if purge_info["horizon_cut_date"] else train_cut
    purged_rows = int((cand["trade_date"] > train_cut).sum())
    purged_horizon_rows = int((cand["trade_date"] > hcut).sum())   # label_end 越界必须删
    embargo_rows = purged_rows - purged_horizon_rows               # 额外隔离多删的

    # ---- 1) train 选参 ----
    cols, signs = _select_composite_params(train_feat, fwd, features, horizon, k)
    alpha_tr = alpha_wide[alpha_wide["trade_date"] <= train_cut]
    best_alpha_col, _ = pick_best_alpha(alpha_tr, fwd, horizon)
    if best_alpha_col is None or not cols:
        return {"error": "train 选参失败", "horizon": horizon}

    # ---- 2) train 端 IC → 融合权重/方向 ----
    comp_tr = _apply_composite(train_feat, cols, signs)
    base_tr = (comp_tr.merge(alpha_wide[["trade_date", "symbol", best_alpha_col]],
                             on=["trade_date", "symbol"], how="inner")
               .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left")
               .dropna(subset=["l2_composite", best_alpha_col, fc]))
    ic_a_tr, _, _ = _rankic_of_column(base_tr, best_alpha_col, fc)
    ic_l_tr, _, _ = _rankic_of_column(base_tr, "l2_composite", fc)
    sign_a = np.sign(ic_a_tr) if not np.isnan(ic_a_tr) and ic_a_tr != 0 else 1.0
    sign_l = np.sign(ic_l_tr) if not np.isnan(ic_l_tr) and ic_l_tr != 0 else 1.0
    w_a = abs(ic_a_tr) if not np.isnan(ic_a_tr) else 0.0
    w_l = abs(ic_l_tr) if not np.isnan(ic_l_tr) else 0.0

    # ---- 3) test 端固定参数评估（同一 stock-day 网格；均按 train 方向定向）----
    comp_te = _apply_composite(test_feat, cols, signs)
    base_te = (comp_te.merge(alpha_wide[["trade_date", "symbol", best_alpha_col]],
                             on=["trade_date", "symbol"], how="inner")
               .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left")
               .dropna(subset=["l2_composite", best_alpha_col, fc]))
    if base_te.empty:
        return {"error": "test 样本不足", "horizon": horizon}

    # alpha 按 train 方向定向（committed direction）；L2 综合分构造时已含 train 方向。
    base_te = base_te.copy()
    base_te["alpha_directed"] = base_te[best_alpha_col] * sign_a
    ev_alpha = _eval_signal(base_te, "alpha_directed", fc)     # train-directed（可承诺的方向）
    ev_alpha_raw = _eval_signal(base_te, best_alpha_col, fc)    # 未定向（透明起见）
    ev_l2 = _eval_signal(base_te, "l2_composite", fc)
    fused = _fuse_column(base_te, best_alpha_col, "l2_composite", w_a, w_l, sign_a, sign_l)
    m_fused = fused.merge(base_te[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left")
    ev_fused = _eval_signal(m_fused, "fused", fc)

    l2_gen = (not np.isnan(ev_l2["rankic"])) and ev_l2["rankic"] > 0
    alpha_gen = (not np.isnan(ev_alpha["rankic"])) and ev_alpha["rankic"] > 0
    fused_gen = (not np.isnan(ev_fused["rankic"])) and ev_fused["rankic"] > 0
    # OOS 名义增量: 融合方向泛化(>0) 且 定向融合 RankIC 高于 定向 alpha。
    inc = bool(fused_gen and not np.isnan(ev_alpha["rankic"])
               and ev_fused["rankic"] > ev_alpha["rankic"] + 1e-6)
    # OOS 稳健增量（同 horizon）: 名义增量 且 L2 综合分自身在 test 方向泛化(>0)。
    # 避免跨 horizon 拼接（如 5d 上 L2 泛化但融合无增量 / 10d 上融合有增量但 L2 未泛化）。
    robust = bool(inc and l2_gen)
    return {
        "horizon": horizon, "train_end": train_end, "test_start": test_start,
        "embargo": emb,
        "purged_train_end": purge_info["last_train_trade_date"],
        "last_train_label_end_date": purge_info["last_train_label_end_date"],
        "first_test_trade_date": purge_info["first_test_trade_date"],
        "purged_rows": purged_rows, "purged_horizon_rows": purged_horizon_rows,
        "embargo_rows": embargo_rows,
        "chosen": cols, "best_alpha": best_alpha_col.replace("a_", ""),
        "sign_a": float(sign_a), "sign_l": float(sign_l),
        "w_a": round(w_a, 4), "w_l": round(w_l, 4),
        "train_ic_alpha": round(ic_a_tr, 4) if not np.isnan(ic_a_tr) else np.nan,
        "train_ic_l2": round(ic_l_tr, 4) if not np.isnan(ic_l_tr) else np.nan,
        "n_train_dates": int(train_feat["trade_date"].nunique()),
        "n_test_dates": int(test_feat["trade_date"].nunique()),
        "test_alpha": ev_alpha, "test_alpha_raw_rankic": round(ev_alpha_raw["rankic"], 4)
        if not np.isnan(ev_alpha_raw["rankic"]) else np.nan,
        "test_l2": ev_l2, "test_fused": ev_fused,
        "alpha_generalizes": bool(alpha_gen), "l2_generalizes": bool(l2_gen),
        "fused_generalizes": bool(fused_gen), "incremental_oos": inc,
        "robust_incremental": robust,
    }
