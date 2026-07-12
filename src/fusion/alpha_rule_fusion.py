"""Alpha191 规则融合核心库（Phase 6A）。

目标：只用 Alpha191 构建多因子规则融合基线，证明「多因子组合是否比最优单因子更稳定」，
而非追求样本内最高收益。Level-2 不参与选因子/方向/权重/正式排序，仅作 shadow 对照。

严格 OOS（无未来函数）：
  - 因子筛选、去相关、方向、权重全部只在 train（≤ purge cut）上确定；
  - 逐 horizon 做 label_end_date < test_start 的 purge，再叠加 embargo；
  - test 期只按 train 冻结的参数套用打分，不看 test 标签。

打分口径：每日截面 rank → z-score → ×方向 → ×权重 → 加权平均（缺失因子按可得权重归一）。
"""
from __future__ import annotations

import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation import factor_validator as fv
from src.validation import level2_validator as lv

PROJECT = Path(__file__).resolve().parent.parent.parent
ALPHA_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191_full"
LABELS = PROJECT / "data" / "processed" / "labels" / "labels.parquet"
DAILY_DIR = PROJECT / "data" / "daily"
L2_FEATURES = PROJECT / "data" / "processed" / "level2" / "level2_daily_features.parquet"

TRAIN_END = "2025-08-31"
TEST_START = "2025-09-01"
EMBARGO = 6                       # horizon purge 之上再额外隔离的交易日数
CORR_CLUSTER_THRESH = 0.70        # 去相关：|每日截面相关| ≥ 此值视为同簇
WEIGHT_CAP = 0.30                 # 单因子权重上限（icir/stability 方案）
COST_ROUNDTRIP_PCT = fv.COST_ROUNDTRIP_PCT   # 多空 spread 往返成本(%)

# --- 候选因子入选门槛（只在 train 上评估；偏宽松以纳入多样候选，再靠去相关精简）---
# 注：扣费后 spread 只作报告，不设硬门槛——单因子长短组合难独立跑赢 4 腿往返成本，
#     否则会退化成 1 个因子，失去多因子融合意义。改用「pre-cost directed_spread>0」
#     作经济一致性 sanity（分位方向须与 IC 方向一致）。
MIN_COVERAGE = 0.80
MIN_ABS_RANKIC = 0.015
MIN_ABS_RANKICIR = 0.25
MIN_MONTHLY_CONSISTENCY = 0.60    # 月度 RankIC 与整体同号的月份占比

FACTORS = [fp.stem for fp in sorted(ALPHA_DIR.glob("signal*.parquet"))]


# ======================================================================
# 数据加载
# ======================================================================
def load_alpha_panel(start: str, end: str) -> pd.DataFrame:
    """把 30 个 Alpha191 信号合成宽表 [trade_date, symbol, signal017..signal046]。"""
    frames = []
    for fp in sorted(ALPHA_DIR.glob("signal*.parquet")):
        s = pd.read_parquet(fp)
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        s = s[(s["trade_date"] >= start) & (s["trade_date"] <= end)].copy()
        s["symbol"] = s["stock_code"].astype(str).str.zfill(6)
        frames.append(s[["trade_date", "symbol", "signal_value"]]
                      .rename(columns={"signal_value": fp.stem}))
    panel = reduce(lambda l, r: l.merge(r, on=["trade_date", "symbol"], how="outer"), frames)
    return panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def load_fwd(horizons=(5, 10)) -> pd.DataFrame:
    """label 表 → fwd_{h}d（open-to-open 原始收益, 百分比）。RankIC/分位对超额与否不敏感。"""
    lab = pd.read_parquet(LABELS)
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["symbol"] = lab["symbol"].astype(str).str.zfill(6)
    out = lab[["trade_date", "symbol"]].copy()
    for h in horizons:
        out[f"fwd_{h}d"] = lab[f"label_{h}d"] * 100
    return out


def load_exposure_controls() -> pd.DataFrame:
    """从日线构建暴露审计控制变量 [trade_date, symbol, log_amount, turnover, log_mktcap, volatility]。

    - log_amount = ln(成交额)               （流动性/规模代理）
    - turnover   = 日线换手率
    - log_mktcap = ln(VWAP × 流通股)，VWAP=成交额/成交量（真实价，非后复权）
    - volatility = 过去 20 日 后复权收益率标准差（仅用 T 及之前，无未来）
    """
    rows = []
    for fp in sorted(DAILY_DIR.glob("*.parquet")):
        code = fp.stem
        if not code.isdigit():
            continue
        d = pd.read_parquet(fp)
        if d.empty or "amount" not in d.columns:
            continue
        d = d.sort_values("date").reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"])
        vol = d["volume"].replace(0, np.nan)
        vwap = d["amount"] / vol
        d["log_amount"] = np.log(d["amount"].where(d["amount"] > 0))
        d["log_mktcap"] = np.log((vwap * d["outstanding_share"]).where(lambda x: x > 0))
        d["volatility"] = d["close"].pct_change().rolling(20, min_periods=10).std()
        d["symbol"] = code.zfill(6)
        rows.append(d[["date", "symbol", "log_amount", "turnover", "log_mktcap", "volatility"]]
                    .rename(columns={"date": "trade_date"}))
    ctrl = pd.concat(rows, ignore_index=True)
    return ctrl


# ======================================================================
# 单因子 train 期打分（RankIC / 稳定性 / 覆盖 / 扣费 spread）
# ======================================================================
def _factor_daily_ic(panel: pd.DataFrame, col: str, fwd: pd.DataFrame, fc: str) -> pd.DataFrame:
    m = (panel[["trade_date", "symbol", col]]
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner")
         .rename(columns={col: "signal_value"}))
    return fv._daily_corr(m, fc)


def screen_factor(panel: pd.DataFrame, col: str, fwd: pd.DataFrame, horizon: int) -> dict:
    """单因子在给定（train）样本上的筛选指标。方向 sign = RankIC 符号。"""
    fc = f"fwd_{horizon}d"
    ic = _factor_daily_ic(panel, col, fwd, fc)
    res = {"factor": col, "coverage": float(panel[col].notna().mean())}
    if not len(ic):
        res.update(rankic=np.nan, rankicir=np.nan, pos_day_ratio=np.nan,
                   monthly_consistency=np.nan, quarterly_consistency=np.nan,
                   directed_spread=np.nan, cost_adj_spread=np.nan, sign=1.0, n_days=0)
        return res
    rankic = float(ic["RankIC"].mean())
    std = float(ic["RankIC"].std())
    sign = 1.0 if rankic >= 0 else -1.0
    # 月度/季度方向稳定性：各月(季)均值与整体 RankIC 同号的占比
    icd = ic.assign(m=ic["trade_date"].dt.to_period("M"), q=ic["trade_date"].dt.to_period("Q"))
    mic = icd.groupby("m")["RankIC"].mean()
    qic = icd.groupby("q")["RankIC"].mean()
    monthly_consistency = float((np.sign(mic) == sign).mean()) if len(mic) else np.nan
    quarterly_consistency = float((np.sign(qic) == sign).mean()) if len(qic) else np.nan
    # 分位 spread（方向对齐后 top−bottom），扣往返成本
    m = (panel[["trade_date", "symbol", col]]
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner")
         .rename(columns={col: "signal_value"}))
    spread, _, _, _ = fv._quintile_spread_turnover(m, fc)
    directed_spread = sign * spread if not np.isnan(spread) else np.nan
    res.update(
        rankic=rankic,
        rankicir=(rankic / std if std > 0 else 0.0),
        pos_day_ratio=float((np.sign(ic["RankIC"]) == sign).mean()),
        monthly_consistency=monthly_consistency,
        quarterly_consistency=quarterly_consistency,
        directed_spread=directed_spread,
        cost_adj_spread=(directed_spread - COST_ROUNDTRIP_PCT if not np.isnan(directed_spread) else np.nan),
        sign=sign, n_days=int(len(ic)),
    )
    return res


def screen_all(panel: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """全 30 因子的 train 筛选表，附 pass（是否满足入池门槛）与 screen_score（代表性排序分）。"""
    rows = [screen_factor(panel, c, fwd, horizon) for c in FACTORS]
    df = pd.DataFrame(rows)
    df["pass"] = (
        (df["coverage"] >= MIN_COVERAGE)
        & (df["rankic"].abs() >= MIN_ABS_RANKIC)
        & (df["rankicir"].abs() >= MIN_ABS_RANKICIR)
        & (df["monthly_consistency"] >= MIN_MONTHLY_CONSISTENCY)
        & (df["directed_spread"] > 0)          # pre-cost 分位方向须与 IC 一致
    )
    # 代表性排序分：优先 RankICIR，其次月度稳定性与覆盖（用于去相关时选代表因子）
    df["screen_score"] = (df["rankicir"].abs()
                          * (0.5 + 0.5 * df["monthly_consistency"].fillna(0))
                          * df["coverage"].fillna(0))
    return df.sort_values("screen_score", ascending=False).reset_index(drop=True)


# ======================================================================
# 去相关（相关簇只留一个代表）
# ======================================================================
def _daily_zrank_matrix(panel: pd.DataFrame, cols: list[str], signs: dict) -> pd.DataFrame:
    """每日截面 rank→z（并按方向对齐），供 pooled 相关近似「日均截面相关」。"""
    parts = []
    for _, g in panel.groupby("trade_date"):
        if len(g) < 10:
            continue
        r = g[cols].rank()
        z = (r - r.mean()) / r.std(ddof=0)
        for c in cols:
            z[c] = z[c] * signs.get(c, 1.0)
        parts.append(z)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def correlation_prune(panel: pd.DataFrame, screen_df: pd.DataFrame,
                      thresh: float = CORR_CLUSTER_THRESH) -> tuple[list[str], pd.DataFrame]:
    """按 screen_score 降序贪心保留：新因子与已留因子的 |日均截面相关| 均 < thresh 才入选。

    返回 (kept, corr_matrix)。相关阵为方向对齐后 z(rank) 的 pooled Pearson（≈日均截面相关）。
    """
    cands = screen_df[screen_df["pass"]].copy()
    cols = cands["factor"].tolist()
    if not cols:
        return [], pd.DataFrame()
    signs = dict(zip(cands["factor"], cands["sign"]))
    zmat = _daily_zrank_matrix(panel, cols, signs)
    corr = zmat.corr(method="pearson")
    kept: list[str] = []
    for c in cols:                                # cands 已按 screen_score 降序
        if all(abs(corr.loc[c, k]) < thresh for k in kept):
            kept.append(c)
    return kept, corr


# ======================================================================
# 四套权重方案（全部只用 train 指标）
# ======================================================================
def _cap_normalize(weights: dict, cap: float) -> dict:
    """归一到 1，并对单因子权重设上限 cap，超限部分迭代重分配给未触顶因子。

    若因子数太少使 n×cap<1（上限不可行），自动放宽 eff_cap=max(cap, 1/n)，保证权重和=1。
    """
    w = {k: max(v, 0.0) for k, v in weights.items()}
    n = len(w)
    if n == 0:
        return {}
    cap = max(cap, 1.0 / n + 1e-9)               # 保证 n×cap ≥ 1（可行）
    total = sum(w.values())
    if total <= 0:
        return {k: 1.0 / n for k in w}
    w = {k: v / total for k, v in w.items()}
    for _ in range(100):
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        room = [k for k in w if w[k] < cap - 1e-12]
        if not room:
            break
        base = sum(w[k] for k in room)
        for k in room:
            w[k] += excess * (w[k] / base if base > 0 else 1.0 / len(room))
    return w


def build_weight_schemes(screen_df: pd.DataFrame, kept: list[str],
                         corr: pd.DataFrame) -> dict:
    """构建 4 套方案的 (factors, signs, weights)。全部基于 train 指标。"""
    sd = screen_df.set_index("factor")
    signs = {c: float(sd.loc[c, "sign"]) for c in kept}

    # 1) best_single：train 最优单因子（screen_score 最高，即 kept[0]）
    best = kept[0]
    best_single = {"factors": [best], "signs": {best: signs[best]}, "weights": {best: 1.0}}

    # 2) equal_weight：去相关候选等权
    equal = {"factors": kept, "signs": signs,
             "weights": {c: 1.0 / len(kept) for c in kept}}

    # 3) icir_weight：权重 ∝ |RankICIR|，单因子上限 WEIGHT_CAP
    icir_raw = {c: abs(float(sd.loc[c, "rankicir"])) for c in kept}
    icir = {"factors": kept, "signs": signs,
            "weights": _cap_normalize(icir_raw, WEIGHT_CAP)}

    # 4) stability_weight：|RankICIR| × 月度稳定 × 覆盖 × 相关惩罚，单因子上限 WEIGHT_CAP
    stab_raw = {}
    for c in kept:
        others = [k for k in kept if k != c]
        avg_abs_corr = float(np.mean([abs(corr.loc[c, k]) for k in others])) if others else 0.0
        stab_raw[c] = (abs(float(sd.loc[c, "rankicir"]))
                       * float(sd.loc[c, "monthly_consistency"] or 0.0)
                       * float(sd.loc[c, "coverage"] or 0.0)
                       * (1.0 - avg_abs_corr))
    stability = {"factors": kept, "signs": signs,
                 "weights": _cap_normalize(stab_raw, WEIGHT_CAP)}

    return {"best_single": best_single, "equal_weight": equal,
            "icir_weight": icir, "stability_weight": stability}


# ======================================================================
# 打分：每日截面 rank→z→×方向→加权平均
# ======================================================================
def build_scheme_scores(panel: pd.DataFrame, scheme: dict) -> pd.DataFrame:
    """套用冻结的 (factors, signs, weights) 到任意面板 → [trade_date, symbol, final_score]。

    缺失因子不计入并按可得权重归一，避免覆盖差异污染量纲。仅用当日截面信息，无未来函数。
    """
    cols = scheme["factors"]
    w = np.array([scheme["weights"][c] for c in cols], dtype=float)
    s = np.array([scheme["signs"][c] for c in cols], dtype=float)
    out = []
    for date, g in panel.groupby("trade_date"):
        if len(g) < 10:
            continue
        R = g[cols].rank().to_numpy(float)                 # 截面 rank，NaN 保留
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 某因子当日全缺→空切片，属正常
            mu = np.nanmean(R, axis=0)
            sd = np.nanstd(R, axis=0)
        sd = np.where(sd == 0, np.nan, sd)
        Z = (R - mu) / sd * s                              # z(rank) × 方向
        present = ~np.isnan(Z)
        Wp = present * w
        denom = Wp.sum(axis=1)
        score = np.where(np.isnan(Z), 0.0, Z) * Wp
        score = score.sum(axis=1) / np.where(denom == 0, np.nan, denom)
        gg = g[["trade_date", "symbol"]].copy()
        gg["final_score"] = score
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["trade_date", "symbol", "final_score"])


# ======================================================================
# 中性化（逐日截面对控制变量 OLS 残差）
# ======================================================================
def neutralize_scores(score_df: pd.DataFrame, controls: pd.DataFrame,
                      ctrl_cols: list[str], value_col: str = "final_score") -> pd.DataFrame:
    """逐日把 value_col 对 controls 做截面 OLS，返回残差列 resid（无控制变量的行剔除）。"""
    m = score_df.merge(controls, on=["trade_date", "symbol"], how="inner").dropna(
        subset=[value_col] + ctrl_cols)
    parts = []
    for date, g in m.groupby("trade_date"):
        if len(g) < len(ctrl_cols) + 5:
            continue
        y = g[value_col].to_numpy(float)
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy(float) for c in ctrl_cols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        gg = g[["trade_date", "symbol"]].copy()
        gg["resid"] = resid
        parts.append(gg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["trade_date", "symbol", "resid"])


# ======================================================================
# 顶层：train 拟合（返回冻结方案 + purge 信息）
# ======================================================================
def purge_train_test(panel: pd.DataFrame, horizon: int,
                     train_end: str = TRAIN_END, test_start: str = TEST_START,
                     embargo: int = EMBARGO):
    """逐 horizon purge：返回 (train_panel, test_panel, purge_info)。train 尾部 label 不越 test_start。"""
    train_cut, info = lv.purge_split_info(panel["trade_date"].unique(),
                                          train_end, test_start, horizon, embargo)
    ts = pd.Timestamp(test_start)
    cand = panel[panel["trade_date"] <= pd.Timestamp(train_end)]
    train_panel = panel[panel["trade_date"] <= train_cut] if train_cut is not None else panel.iloc[0:0]
    test_panel = panel[panel["trade_date"] >= ts]
    hcut = pd.Timestamp(info["horizon_cut_date"]) if info["horizon_cut_date"] else train_cut
    info["purged_rows"] = int((cand["trade_date"] > train_cut).sum()) if train_cut is not None else 0
    info["purged_horizon_rows"] = int((cand["trade_date"] > hcut).sum()) if hcut is not None else 0
    info["embargo_rows"] = info["purged_rows"] - info["purged_horizon_rows"]
    return train_panel, test_panel, info


def fit_fusion(panel: pd.DataFrame, fwd: pd.DataFrame, horizon: int,
               train_end: str = TRAIN_END, test_start: str = TEST_START,
               embargo: int = EMBARGO) -> dict:
    """在 train 上完成筛选→去相关→四方案权重，返回冻结方案与元信息（不接触 test 标签）。"""
    train_panel, _, info = purge_train_test(panel, horizon, train_end, test_start, embargo)
    screen_df = screen_all(train_panel, fwd, horizon)
    kept, corr = correlation_prune(train_panel, screen_df)
    if not kept:
        return {"error": "无因子通过筛选", "horizon": horizon, "screen": screen_df, "purge": info}
    schemes = build_weight_schemes(screen_df, kept, corr)
    return {"horizon": horizon, "schemes": schemes, "kept": kept,
            "screen": screen_df, "corr": corr, "purge": info,
            "train_cut": info.get("last_train_trade_date")}
