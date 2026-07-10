"""Level-2 特征验证器（Phase 5）。

复用 Alpha191 验证管线的核心函数（factor_validator._daily_corr /
_quintile_spread_turnover），但改用 **open-to-open 超额收益**（label_*d_excess_index）
作为 fwd，在 **Universe_C**（有 Level-2 数据的股票）上验证每个 Level-2 特征。

产出三类证据（暂不做 ML）:
  1. 单特征稳定性: 各 horizon(1/3/5/10d) 的 RankIC / RankICIR / spread / 覆盖。
  2. 与 Alpha191 的正交性: |相关| 越低越说明信息不重叠。
  3. 增量证明（无 ML）: 在同一批 Level-2 stock-day 上，比较
       RankIC(Alpha191 最优) vs RankIC(Level-2 最优) vs RankIC(等权 rank 融合)，
     若融合 > 单独，即 Level-2 对 Alpha191 有增量。

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
        spread, topq, botq, _ = fv._quintile_spread_turnover(merged, fc)
        res[f"spread_{h}d"] = spread
        res[f"cost_adj_spread_{h}d"] = spread - COST if not np.isnan(spread) else np.nan
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


def orthogonality(feat_df: pd.DataFrame, alpha_wide: pd.DataFrame,
                  l2_feature: str, top_k: int = 8) -> pd.DataFrame:
    """Level-2 特征与各 Alpha191 的截面 Spearman 相关（越低越正交）。返回 |corr| 最大的 top_k。"""
    m = feat_df[["trade_date", "symbol", l2_feature]].merge(
        alpha_wide, on=["trade_date", "symbol"], how="inner")
    a_cols = [c for c in alpha_wide.columns if c.startswith("a_")]
    rows = []
    for a in a_cols:
        v = m[[l2_feature, a]].dropna()
        if len(v) < 30:
            continue
        rows.append({"alpha": a.replace("a_", ""),
                     "spearman": round(float(v[l2_feature].corr(v[a], method="spearman")), 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs"] = out["spearman"].abs()
    return out.sort_values("abs", ascending=False).head(top_k).drop(columns="abs").reset_index(drop=True)


def build_l2_composite(feat_df: pd.DataFrame, fwd: pd.DataFrame, features: list[str],
                       horizon: int = 5, k: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """把 top-k 个（按 |RankICIR_{h}d|）Level-2 特征做**符号对齐的等权 z 分**合成一个综合分。

    弱但正交的多个特征聚合可提取增量。返回 (含 l2_composite 列的表, 入选特征名)。
    注: top-k 选择与符号对齐用全样本 IC，属**样本内**方向性证据（OOS 留待 Phase 9）。
    """
    fc = f"fwd_{horizon}d"
    scored = []
    for c in features:
        r = validate_feature(feat_df, fwd, c)
        scored.append((c, r.get(f"RankIC_{horizon}d", np.nan), r.get(f"RankICIR_{horizon}d", np.nan)))
    scored = [s for s in scored if not np.isnan(s[1])]
    scored.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else 0, reverse=True)
    chosen = scored[:k]
    signs = {c: (1.0 if ic >= 0 else -1.0) for c, ic, _ in chosen}
    cols = [c for c, _, _ in chosen]

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
        g["l2_composite"] = z_sum / len(cols)
        parts.append(g[["trade_date", "symbol", "l2_composite"]])
    comp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["trade_date", "symbol", "l2_composite"])
    return comp, cols


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
