"""Alpha191 因子验证器（在 Universe_A/B/C 上重跑）。

用无未来函数的 open-to-open label（label_builder）计算:
- RankIC / ICIR / RankICIR（每 horizon）
- 分月 RankIC、分市值组 RankIC、分流动性组 RankIC（5d）
- 五分位收益、多空 spread、扣费后 spread
- Top 分位换手率
- 覆盖率、缺失率

分行业/分年度不可用（无行业数据、单一年份），标注 N/A / 降级为分季度。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
LABELS = PROJECT / "data" / "processed" / "labels" / "labels.parquet"

HORIZONS = [1, 3, 5, 10]
N_Q = 5
# 单边成本 = 佣金0.0003 + 滑点0.001 = 0.0013(0.13%)。
# 多空 spread 含 4 个单边(多头买卖 + 空头买卖) → 往返 0.52%。
COST_ROUNDTRIP_PCT = 4 * (0.0003 + 0.001) * 100  # 0.52%


def load_labels_as_fwd() -> pd.DataFrame:
    """label 表转为 fwd_{h}d（百分比），供 IC/分位计算。RankIC 对是否超额不敏感。"""
    lab = pd.read_parquet(LABELS)
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["symbol"] = lab["symbol"].astype(str).str.zfill(6)
    out = lab[["trade_date", "symbol"]].copy()
    for h in HORIZONS:
        out[f"fwd_{h}d"] = lab[f"label_{h}d"] * 100
    return out


def _load_signal(sig_path: Path, start: str, end: str) -> pd.DataFrame:
    s = pd.read_parquet(sig_path)
    s["trade_date"] = pd.to_datetime(s["trade_date"])
    s = s[(s["trade_date"] >= start) & (s["trade_date"] <= end)].copy()
    s["symbol"] = s["stock_code"].astype(str).str.zfill(6)
    return s[["trade_date", "symbol", "signal_value", "signal_id", "signal_name"]]


def _daily_corr(merged: pd.DataFrame, fwd_col: str) -> pd.DataFrame:
    """每日 IC(pearson) 与 RankIC(spearman)。"""
    rows = []
    for date, g in merged.groupby("trade_date"):
        v = g[["signal_value", fwd_col]].dropna()
        if len(v) < 5:
            continue
        rows.append({
            "trade_date": date,
            "IC": v["signal_value"].corr(v[fwd_col], method="pearson"),
            "RankIC": v["signal_value"].corr(v[fwd_col], method="spearman"),
        })
    return pd.DataFrame(rows)


def _quintile_spread_turnover(merged: pd.DataFrame, fwd_col: str) -> tuple[float, float, float, list]:
    """返回 (spread%, top_q_mean%, bottom_q_mean%, top_membership_by_date)。"""
    top_ret, bot_ret = [], []
    top_sets = []
    for date, g in merged.groupby("trade_date"):
        gc = g.dropna(subset=["signal_value", fwd_col])
        if len(gc) < N_Q * 2:
            continue
        try:
            q = pd.qcut(gc["signal_value"], N_Q, labels=False, duplicates="drop")
        except ValueError:
            continue
        gc = gc.assign(q=q)
        qmeans = gc.groupby("q")[fwd_col].mean()
        if 0 in qmeans.index and (N_Q - 1) in qmeans.index:
            top_ret.append(qmeans[N_Q - 1])
            bot_ret.append(qmeans[0])
            top_sets.append((date, set(gc[gc["q"] == N_Q - 1]["symbol"])))
    if not top_ret:
        return np.nan, np.nan, np.nan, []
    spread = np.mean(top_ret) - np.mean(bot_ret)
    return spread, float(np.mean(top_ret)), float(np.mean(bot_ret)), top_sets


def _turnover(top_sets: list) -> float:
    """相邻日 top 分位成员变动比例的均值。"""
    if len(top_sets) < 2:
        return np.nan
    tos = []
    for (_, s0), (_, s1) in zip(top_sets[:-1], top_sets[1:]):
        if not s0:
            continue
        tos.append(1 - len(s0 & s1) / len(s0))
    return float(np.mean(tos)) if tos else np.nan


def _grouped_rankic(merged: pd.DataFrame, fwd_col: str, meta: pd.DataFrame, group_col: str, n_groups: int = 3) -> dict:
    """按 meta[group_col] 分组，各组日均 RankIC（5d）。"""
    m = merged.merge(meta[["symbol", group_col]], on="symbol", how="left")
    try:
        m["grp"] = pd.qcut(m[group_col], n_groups, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return {}
    out = {}
    for grp, gg in m.groupby("grp", observed=True):
        ic = _daily_corr(gg, fwd_col)
        out[str(grp)] = round(float(ic["RankIC"].mean()), 4) if len(ic) else np.nan
    return out


def validate_factor(
    sig_path: Path, fwd: pd.DataFrame, universe_syms: set, meta: pd.DataFrame,
    start: str, end: str,
) -> dict:
    """在单个 universe 上验证单个因子。"""
    sig = _load_signal(sig_path, start, end)
    sig = sig[sig["symbol"].isin(universe_syms)]
    sid = sig["signal_id"].iloc[0] if len(sig) else sig_path.stem
    sname = sig["signal_name"].iloc[0] if len(sig) else ""
    merged = sig.merge(fwd, on=["trade_date", "symbol"], how="left")

    res = {"signal_id": sid, "signal_name": sname,
           "n_stocks": sig["symbol"].nunique(), "n_dates": sig["trade_date"].nunique(),
           "missing_pct": sig["signal_value"].isna().mean() * 100}

    for h in HORIZONS:
        fc = f"fwd_{h}d"
        res[f"coverage_{h}d"] = merged[fc].notna().mean() * 100
        ic = _daily_corr(merged, fc)
        if len(ic):
            res[f"RankIC_{h}d"] = ic["RankIC"].mean()
            res[f"RankICIR_{h}d"] = ic["RankIC"].mean() / ic["RankIC"].std() if ic["RankIC"].std() > 0 else 0.0
            res[f"IC_{h}d"] = ic["IC"].mean()
        else:
            res[f"RankIC_{h}d"] = res[f"RankICIR_{h}d"] = res[f"IC_{h}d"] = np.nan
        spread, topq, botq, top_sets = _quintile_spread_turnover(merged, fc)
        res[f"spread_{h}d"] = spread
        res[f"cost_adj_spread_{h}d"] = spread - COST_ROUNDTRIP_PCT if not np.isnan(spread) else np.nan
        if h == 5:
            res["turnover_5d"] = _turnover(top_sets)

    # 分季度 RankIC (5d) — 替代分年度
    ic5 = _daily_corr(merged, "fwd_5d")
    if len(ic5):
        ic5 = ic5.copy()
        ic5["q"] = ic5["trade_date"].dt.quarter
        qic = ic5.groupby("q")["RankIC"].mean()
        res["quarters_positive_5d"] = int((qic > 0).sum())
        res["quarterly_rankic_5d"] = {int(k): round(v, 4) for k, v in qic.items()}
    else:
        res["quarters_positive_5d"] = 0
        res["quarterly_rankic_5d"] = {}

    # 分月 RankIC (5d)
    if len(ic5):
        mic = ic5.assign(m=ic5["trade_date"].dt.month).groupby("m")["RankIC"].mean()
        res["monthly_rankic_5d"] = {int(k): round(v, 4) for k, v in mic.items()}
    else:
        res["monthly_rankic_5d"] = {}

    # 分市值/流动性组 RankIC (5d)
    res["mktcap_rankic_5d"] = _grouped_rankic(merged, "fwd_5d", meta, "market_cap_est")
    res["liq_rankic_5d"] = _grouped_rankic(merged, "fwd_5d", meta, "median_amount")

    return res


def recommend(row_B: dict) -> tuple[str, str]:
    """基于主池 Universe_B 的保留决策。"""
    rk = row_B.get("RankIC_5d", np.nan)
    rkir = row_B.get("RankICIR_5d", np.nan)
    cadj = row_B.get("cost_adj_spread_5d", np.nan)
    qpos = row_B.get("quarters_positive_5d", 0)
    cov = row_B.get("coverage_5d", 0)
    reasons = []
    keep = True
    if not (abs(rk) > 0.015):
        keep = False; reasons.append(f"|RankIC_5d|={abs(rk):.3f}≤0.015")
    if not (abs(rkir) > 0.30):
        keep = False; reasons.append(f"|RankICIR_5d|={abs(rkir):.2f}≤0.30")
    if not (np.sign(rk) * cadj > 0 if not np.isnan(cadj) else False):
        keep = False; reasons.append(f"扣费后 spread 方向不利({cadj:+.2f}%)")
    if qpos < 2 and rk > 0:
        keep = False; reasons.append(f"仅{qpos}季度 RankIC>0")
    if cov < 80:
        keep = False; reasons.append(f"覆盖率{cov:.0f}%<80%")
    if keep:
        tag = "重点关注" if abs(rkir) > 0.50 else "保留"
        direction = "反向" if rk < 0 else "正向"
        return tag, f"{direction}有效; " + (", ".join(reasons) if reasons else "全部达标")
    return "淘汰", "; ".join(reasons)
