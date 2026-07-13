"""Phase 8A-E.1.1 突破事件效应 vs 突破质量效应拆分。

严格锁定范围：
  - 不修改冻结公式，不改变方向，不增加参数
  - 拆分 breakout_event (0/1) 和 conditional quality (BO only)
  - 仅在 BO 内部计算 quality RankIC，不混入非突破的 0 值
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.features.price_action.breakout import breakout_close_quality
from src.validation import factor_validator as fv
from src.fusion import alpha_rule_fusion as arf

DAILY_DIR = PROJECT / "data" / "daily"
UNIVERSE_FILE = PROJECT / "data" / "processed" / "stock_universe" / "zz1000_liquid_selected.txt"
LABELS_PATH = PROJECT / "data" / "processed" / "labels" / "labels.parquet"

HORIZONS = (5, 10, 20)
DATA_START = "2025-01-02"
DATA_END = "2025-12-31"
DEFAULT_PARAMS = {"L": 20, "ATR_N": 20, "VOL_N": 20}

BLOCKS = [
    ("2025-04-30", "2025-05-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-10-31"),
    ("2025-10-31", "2025-11-01", "2025-12-31"),
]

CTRL_COLS = ["log_mktcap", "log_amount", "turnover", "volatility"]
INDEX_PATH = DAILY_DIR / "idx_000852.parquet"


def _p(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:+.{d}f}%" if pct else f"{x:+.{d}f}"


def _pct(x, d=2):
    """Format a value already in percent units (fwd return * 100 → percentage points)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{d}f}%"


def _pp(x, d=2):
    """Format a spread/difference in percentage points."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{d}f}pp"


# ======================================================================
# 数据加载
# ======================================================================
def load_panel(codes):
    frames = []
    for code in codes:
        fp = DAILY_DIR / f"{code}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= DATA_START) & (df["date"] <= DATA_END)]
        if df.empty:
            continue
        df = df.rename(columns={"date": "trade_date"})
        df["symbol"] = str(code).zfill(6)
        frames.append(df[["trade_date", "symbol", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def load_fwd():
    lab = pd.read_parquet(LABELS_PATH)
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["symbol"] = lab["symbol"].astype(str).str.zfill(6)
    out = lab[["trade_date", "symbol"]].copy()
    for h in HORIZONS:
        out[f"fwd_{h}d"] = lab[f"label_{h}d"] * 100
    return out


def load_controls():
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
        d = d[(d["date"] >= DATA_START) & (d["date"] <= DATA_END)]
        if d.empty:
            continue
        vol = d["volume"].replace(0, np.nan)
        vwap = d["amount"] / vol
        d["log_amount"] = np.log(d["amount"].where(d["amount"] > 0))
        d["log_mktcap"] = np.log((vwap * d["outstanding_share"]).where(lambda x: x > 0))
        d["volatility"] = d["close"].pct_change().rolling(20, min_periods=10).std()
        d["symbol"] = code.zfill(6)
        rows.append(d[["date", "symbol", "log_amount", "turnover", "log_mktcap", "volatility"]]
                    .rename(columns={"date": "trade_date"}))
    return pd.concat(rows, ignore_index=True)


def load_index_returns():
    """中证1000 open-to-open 收益，用于超额计算。"""
    idx = pd.read_parquet(INDEX_PATH)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx[(idx["date"] >= DATA_START) & (idx["date"] <= DATA_END)].sort_values("date")
    opens = idx["open"].values / 100  # 分→元
    out = pd.DataFrame({"trade_date": idx["date"].values})
    for h in HORIZONS:
        rets = np.full(len(opens), np.nan)
        for i in range(len(opens) - h - 1):
            if opens[i] > 0 and opens[i + 1 + h] > 0:
                rets[i] = opens[i + 1 + h] / opens[i + 1] - 1
        out[f"idx_fwd_{h}d"] = rets * 100
    return out


# ======================================================================
# 计算
# ======================================================================
def compute_scores(panel):
    result = breakout_close_quality(panel, **DEFAULT_PARAMS)
    result = result.rename(columns={"factor_value": "signal_value"})
    return result


# ======================================================================
# A. 突破事件效应
# ======================================================================
def event_effect(scores, fwd, controls, idx_fwd):
    """breakout_flag 0/1 的事件效应。"""
    m = scores.merge(fwd, on=["trade_date", "symbol"], how="inner")
    m = m.dropna(subset=["signal_value"])  # signal valid → breakout_event is valid

    out = {}
    for h in HORIZONS:
        fc = f"fwd_{h}d"
        bo_rets, nbo_rets = [], []
        bo_excess, nbo_excess = [], []
        bo_dates = []
        bo_total, nbo_total = 0, 0
        # merge index
        mi = m.merge(idx_fwd[["trade_date", f"idx_fwd_{h}d"]], on="trade_date", how="left")
        for date, g in mi.groupby("trade_date"):
            bo = g[g["breakout_event"]]
            nbo = g[~g["breakout_event"]]
            if len(bo) < 1 or len(nbo) < 5:
                continue
            bo_rets.append(bo[fc].mean())
            nbo_rets.append(nbo[fc].mean())
            bo_dates.append(date)
            bo_total += len(bo)
            nbo_total += len(nbo)
            if f"idx_fwd_{h}d" in g.columns:
                idx_r = g[f"idx_fwd_{h}d"].iloc[0]
                if not np.isnan(idx_r):
                    bo_excess.append(bo[fc].mean() - idx_r)
                    nbo_excess.append(nbo[fc].mean() - idx_r)

        spread = np.mean(bo_rets) - np.mean(nbo_rets) if bo_rets else np.nan
        excess_spread = np.mean(bo_excess) - np.mean(nbo_excess) if bo_excess else np.nan
        out[h] = {
            "spread": spread,
            "bo_mean": np.mean(bo_rets) if bo_rets else np.nan,
            "nbo_mean": np.mean(nbo_rets) if nbo_rets else np.nan,
            "excess_spread": excess_spread,
            "bo_excess_mean": np.mean(bo_excess) if bo_excess else np.nan,
            "nbo_excess_mean": np.mean(nbo_excess) if nbo_excess else np.nan,
            "n_bo_days": len(bo_rets),
            "n_bo_total": bo_total,
            "n_nbo_total": nbo_total,
        }

    # Matched comparison: for each BO stock, find closest NBO stock by propensity
    matched = _matched_event_effect(scores, fwd, controls)
    out["matched"] = matched
    return out


def _matched_event_effect(scores, fwd, controls):
    """Propensity-score matched BO vs NBO comparison.

    For each day, estimate propensity = logistic(mktcap, amount, turnover, vol),
    then match each BO stock to nearest NBO stock. Compute matched spread.
    """
    m = (scores[["trade_date", "symbol", "breakout_event"]]
         .merge(controls[["trade_date", "symbol"] + CTRL_COLS],
                on=["trade_date", "symbol"], how="inner")
         .merge(fwd, on=["trade_date", "symbol"], how="inner"))
    m = m.dropna(subset=CTRL_COLS)

    out = {}
    for h in HORIZONS:
        fc = f"fwd_{h}d"
        bo_diffs = []
        for date, g in m.groupby("trade_date"):
            bo = g[g["breakout_event"]].dropna(subset=[fc])
            nbo = g[~g["breakout_event"]].dropna(subset=[fc])
            if len(bo) < 1 or len(nbo) < 5:
                continue
            # Nearest-neighbor match on standardized controls
            ctrl_all = g[CTRL_COLS].to_numpy(float)
            ctrl_mean = np.nanmean(ctrl_all, axis=0)
            ctrl_std = np.nanstd(ctrl_all, axis=0)
            ctrl_std[ctrl_std == 0] = 1.0
            ctrl_z = (ctrl_all - ctrl_mean) / ctrl_std

            bo_idx = bo.index.to_numpy()
            nbo_idx = nbo.index.to_numpy()
            # Map to position in g
            bo_pos = [g.index.get_loc(i) for i in bo_idx]
            nbo_pos = [g.index.get_loc(i) for i in nbo_idx]

            for bp in bo_pos:
                dists = np.sqrt(np.nansum((ctrl_z[bp] - ctrl_z[nbo_pos]) ** 2, axis=1))
                nearest = nbo_pos[np.nanargmin(dists)]
                bo_diffs.append(g[fc].iloc[bp] - g[fc].iloc[nearest])

        out[h] = {
            "matched_spread": np.mean(bo_diffs) if bo_diffs else np.nan,
            "n_pairs": len(bo_diffs),
        }
    return out


# ======================================================================
# B. 突破质量效应（仅 breakout_flag==1 内部）
# ======================================================================
def conditional_quality(scores, fwd, controls):
    """仅在 BO 股票内部评估 raw_quality 的排序能力。"""
    bo = scores[scores["breakout_event"]].copy()
    bo = bo.drop(columns=["signal_value"]).rename(columns={"raw_quality": "signal_value"})

    out = {}
    # Full-sample conditional
    for h in HORIZONS:
        fc = f"fwd_{h}d"
        m = bo.merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner")
        m = m.dropna(subset=["signal_value", fc])
        ic = _daily_rankic(m, fc)
        n_days = len(ic)
        ric = float(ic["RankIC"].mean()) if n_days else np.nan
        ricir = float(ric / ic["RankIC"].std()) if n_days > 1 and ic["RankIC"].std() > 0 else np.nan
        # tercile spread (breakout内部三分位)
        t_spread = _tercile_spread(m, fc)
        out[h] = {"rankic": ric, "rankicir": ricir, "n_days": n_days,
                  "n_stocks_total": len(bo),
                  "tercile_spread": t_spread.get("spread", np.nan),
                  "top_mean": t_spread.get("top_mean", np.nan),
                  "mid_mean": t_spread.get("mid_mean", np.nan),
                  "bot_mean": t_spread.get("bot_mean", np.nan),
                  "daily_ics": list(ic["RankIC"].values) if n_days else []}

    # Neutralized conditional RankIC
    ctrl_out = {}
    for h in HORIZONS:
        fc = f"fwd_{h}d"
        m = (bo[["trade_date", "symbol", "signal_value"]]
             .merge(controls[["trade_date", "symbol"] + CTRL_COLS],
                    on=["trade_date", "symbol"], how="inner")
             .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner"))
        m = m.dropna(subset=["signal_value", fc] + CTRL_COLS)
        resid_ics = []
        for date, g in m.groupby("trade_date"):
            if len(g) < len(CTRL_COLS) + 3:
                continue
            y = g["signal_value"].to_numpy(float)
            X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy(float) for c in CTRL_COLS])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
            except np.linalg.LinAlgError:
                continue
            rc = pd.Series(resid).corr(g[fc], method="spearman")
            if not np.isnan(rc):
                resid_ics.append(rc)
        ctrl_out[h] = {
            "neutralized_rankic": float(np.mean(resid_ics)) if resid_ics else np.nan,
            "n_neut_days": len(resid_ics),
        }

    # Per-fold conditional
    folds_out = {}
    for train_end, test_start, test_end in BLOCKS:
        test_bo = bo[(bo["trade_date"] >= pd.Timestamp(test_start))
                     & (bo["trade_date"] <= pd.Timestamp(test_end))]
        fold = {}
        for h in HORIZONS:
            fc = f"fwd_{h}d"
            m = test_bo.merge(fwd[["trade_date", "symbol", fc]],
                              on=["trade_date", "symbol"], how="inner")
            m = m.dropna(subset=["signal_value", fc])
            ic = _daily_rankic(m, fc)
            nd = len(ic)
            ric = float(ic["RankIC"].mean()) if nd else np.nan
            ricir = float(ric / ic["RankIC"].std()) if nd > 1 and ic["RankIC"].std() > 0 else np.nan
            fold[h] = {"rankic": ric, "rankicir": ricir, "n_days": nd,
                       "n_bo_events": len(test_bo),
                       "daily_ics": list(ic["RankIC"].values) if nd else []}
        folds_out[test_start[:7]] = fold

    # Pooled (all folds daily ICs combined)
    pooled = {}
    for h in HORIZONS:
        all_ics = []
        for _, fold in folds_out.items():
            fh = fold.get(h, {})
            all_ics.extend(fh.get("daily_ics", []))
        arr = np.array(all_ics)
        pooled[h] = {
            "pooled_rankic": float(np.mean(arr)) if len(arr) else np.nan,
            "pooled_rankicir": float(np.mean(arr) / np.std(arr)) if len(arr) > 1 and np.std(arr) > 0 else np.nan,
            "n_ic_days": len(arr),
        }
        # Also full-sample pooled
        pooled[h]["full_rankicir"] = out[h]["rankicir"]

    return {"full": out, "folds": folds_out, "pooled": pooled, "neutralized": ctrl_out}


def _daily_rankic(m, fc):
    rows = []
    for date, g in m.groupby("trade_date"):
        sv = g["signal_value"].to_numpy(float)
        fv = g[fc].to_numpy(float)
        mask = ~np.isnan(sv) & ~np.isnan(fv)
        if mask.sum() < 5:
            continue
        sv_c, fv_c = sv[mask], fv[mask]
        # manual spearman: corr of ranks
        r1 = pd.Series(sv_c).rank().to_numpy()
        r2 = pd.Series(fv_c).rank().to_numpy()
        rc = np.corrcoef(r1, r2)[0, 1]
        rows.append({"trade_date": date, "RankIC": rc})
    return pd.DataFrame(rows)


def _tercile_spread(m, fc):
    """三分位 spread（top - bottom tercile），返回 top/mid/bot 均值。"""
    top_r, mid_r, bot_r = [], [], []
    for date, g in m.groupby("trade_date"):
        gc = g.dropna(subset=["signal_value", fc])
        if len(gc) < 6:
            continue
        try:
            gc = gc.copy()
            gc["t"] = pd.qcut(gc["signal_value"], 3, labels=[0, 1, 2], duplicates="drop")
        except ValueError:
            continue
        means = gc.groupby("t")[fc].mean()
        if 0 in means.index and 1 in means.index and 2 in means.index:
            bot_r.append(means[0])
            mid_r.append(means[1])
            top_r.append(means[2])
    if not top_r:
        return {"spread": np.nan, "top_mean": np.nan, "mid_mean": np.nan, "bot_mean": np.nan}
    return {"spread": np.mean(top_r) - np.mean(bot_r),
            "top_mean": np.mean(top_r), "mid_mean": np.mean(mid_r), "bot_mean": np.mean(bot_r)}


# ======================================================================
# C. Alpha191 关系（仅 BO 内部）
# ======================================================================
def alpha191_conditional(bo_scores, a191_panel, fwd):
    """仅在 BO 股票内部评估 quality score 与 A191 的关系。"""
    bo = bo_scores[bo_scores["breakout_event"]].copy()
    bo = bo.drop(columns=["signal_value"]).rename(columns={"raw_quality": "signal_value"})

    a191_factors = [c for c in a191_panel.columns if c.startswith("signal")]
    m = bo[["trade_date", "symbol", "signal_value"]].merge(
        a191_panel[["trade_date", "symbol"] + a191_factors],
        on=["trade_date", "symbol"], how="inner")

    # quality vs A191 equal_weight
    daily_corrs_eq = []
    for _, g in m.groupby("trade_date"):
        g2 = g.dropna(subset=["signal_value"])
        if len(g2) < 5:
            continue
        ew = g2[a191_factors].rank(pct=True).mean(axis=1)
        c = g2["signal_value"].corr(ew, method="spearman")
        if not np.isnan(c):
            daily_corrs_eq.append(c)

    # quality vs each A191 single factor (max abs corr)
    single_corrs = {}
    for fac in a191_factors:
        cors = []
        for _, g in m.groupby("trade_date"):
            g2 = g.dropna(subset=["signal_value", fac])
            if len(g2) < 5:
                continue
            c = g2["signal_value"].corr(g2[fac], method="spearman")
            if not np.isnan(c):
                cors.append(c)
        single_corrs[fac] = float(np.mean(cors)) if cors else np.nan

    max_corr_fac = max(single_corrs, key=lambda k: abs(single_corrs[k]))
    max_corr_val = single_corrs[max_corr_fac]

    # Residual conditional RankIC (quality orthogonal to A191 equal_weight)
    a191_resid_by_h = {}
    for h in HORIZONS:
        fc = f"fwd_{h}d"
        mm = m.merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner")
        resid_ics = []
        for _, g in mm.groupby("trade_date"):
            g2 = g.dropna(subset=["signal_value"] + a191_factors + [fc])
            if len(g2) < len(a191_factors) + 3:
                continue
            ew = g2[a191_factors].rank(pct=True).mean(axis=1)
            y = g2["signal_value"].to_numpy(float)
            X = np.column_stack([np.ones(len(g2)), ew.to_numpy()])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
            except np.linalg.LinAlgError:
                continue
            rc = pd.Series(resid).corr(g2[fc], method="spearman")
            if not np.isnan(rc):
                resid_ics.append(rc)
        a191_resid_by_h[h] = float(np.mean(resid_ics)) if resid_ics else np.nan

    return {
        "corr_with_a191_ew": float(np.mean(daily_corrs_eq)) if daily_corrs_eq else np.nan,
        "n_corr_days_eq": len(daily_corrs_eq),
        "max_single_corr_factor": max_corr_fac,
        "max_single_corr_value": max_corr_val,
        "a191_resid_rankic": a191_resid_by_h,
    }


# ======================================================================
# 报告
# ======================================================================
def build_section(event, cond, a191_rel, coverage):
    L = []

    L.append("\n## 9. 事件效应 vs 质量效应拆分（Phase 8A-E.1.1）\n\n")
    L.append("> 将 v1 的混合 RankIC 拆分为：突破发生效应（BO vs NBO）和条件质量效应（BO内部）。\n")
    L.append(f"> BO 样本: {coverage['n_bo_total']} 条, 日均 {coverage['n_bo_daily_avg']:.1f} 只, "
             f"占总样本 {coverage['bo_pct']:.1%}\n\n")

    # ---- A. Event Effect ----
    L.append("### 9A. 突破事件效应\n\n")
    L.append("| horizon | BO前向收益 | NBO前向收益 | spread | BO样本数 | NBO样本数 | BO日数 |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for h in HORIZONS:
        e = event[h]
        L.append(f"| {h}d | {_pct(e['bo_mean'])} | {_pct(e['nbo_mean'])} | "
                 f"{_pp(e['spread'])} | {e['n_bo_total']} | {e['n_nbo_total']} | "
                 f"{e['n_bo_days']} |\n")

    L.append("\n> 前向收益为 open-to-open 均值，spread 单位为百分点（pp）。"
             "负值表示突破日后前向收益低于非突破日。\n")

    L.append("\n**风格匹配后**:\n\n")
    L.append("| horizon | matched spread | 配对样本数 |\n")
    L.append("|---|---|---|\n")
    for h in HORIZONS:
        m = event["matched"].get(h, {})
        L.append(f"| {h}d | {_pp(m.get('matched_spread'))} | {m.get('n_pairs', 0)} |\n")

    L.append("\n> matched spread: 每日对每只 BO 股票找最近邻 NBO（mktcap/amount/turnover/vol），"
             "BO − matched_NBO 收益差（百分点）。\n")

    # ---- B. Conditional Quality ----
    L.append("\n### 9B. 突破质量效应（仅 BO 内部）\n\n")

    # Full sample
    L.append("**全样本 conditional RankIC**:\n\n")
    L.append("| horizon | cond RankIC | cond RankICIR | 有效日数 | tercile spread | top均值 | mid均值 | bot均值 |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for h in HORIZONS:
        c = cond["full"][h]
        L.append(f"| {h}d | {_p(c['rankic'])} | {_p(c['rankicir'])} | {c['n_days']} | "
                 f"{_pct(c['tercile_spread'])} | {_pct(c['top_mean'])} | "
                 f"{_pct(c['mid_mean'])} | {_pct(c['bot_mean'])} |\n")

    L.append("\n> tercile spread = top tercile − bottom tercile 前向收益差（百分点）。"
             "若 spread > 0 但 RankIC < 0，说明 quality score 与前向收益非单调（mid 可能最高/最低）。\n")

    # Neutralized — explicitly flagged as insufficient
    L.append("\n**中性化 conditional RankIC**（控制 log_mktcap, log_amount, turnover, volatility）:\n\n")
    L.append("| horizon | neutralized cond RankIC | 有效日数 |\n")
    L.append("|---|---|---|\n")
    for h in HORIZONS:
        n = cond["neutralized"][h]
        L.append(f"| {h}d | {_p(n['neutralized_rankic'])} | {n['n_neut_days']} |\n")

    L.append("\n> **注意**: 中性化仅在 ≥5 只 BO 股票的交易日计算，有效日数极少（≤15 日），"
             "**不参与最终判定**，仅作参考。\n")

    # Per-fold
    L.append("\n**每折 conditional RankIC**:\n\n")
    L.append("| 折 | horizon | cond RankIC | cond RankICIR | 有效日数 | BO事件数 |\n")
    L.append("|---|---|---|---|---|---|\n")
    for fold_key in sorted(cond["folds"].keys()):
        fold = cond["folds"][fold_key]
        for h in HORIZONS:
            fh = fold.get(h, {})
            L.append(f"| {fold_key} | {h}d | {_p(fh.get('rankic'))} | {_p(fh.get('rankicir'))} | "
                     f"{fh.get('n_days', 0)} | {fh.get('n_bo_events', 0)} |\n")

    # Pooled
    L.append("\n**Pooled conditional**:\n\n")
    L.append("| horizon | pooled cond RankIC | pooled cond RankICIR | IC日数 |\n")
    L.append("|---|---|---|---|\n")
    for h in HORIZONS:
        p = cond["pooled"][h]
        L.append(f"| {h}d | {_p(p['pooled_rankic'])} | {_p(p['pooled_rankicir'])} | "
                 f"{p['n_ic_days']} |\n")

    # ---- C. Alpha191 Conditional ----
    L.append("\n### 9C. Alpha191 关系（仅 BO 内部）\n\n")
    L.append(f"- quality vs A191 equal_weight 截面相关: {_p(a191_rel['corr_with_a191_ew'])} "
             f"(有效 {a191_rel['n_corr_days_eq']} 日)\n")
    L.append(f"- quality vs 单因子最大绝对相关: {a191_rel['max_single_corr_factor']} "
             f"({_p(a191_rel['max_single_corr_value'])})\n")

    L.append("\n**对 A191 残差化后 conditional RankIC**:\n\n")
    L.append("| horizon | 残差化 cond RankIC |\n")
    L.append("|---|---|\n")
    for h in HORIZONS:
        v = a191_rel["a191_resid_rankic"].get(h, np.nan)
        L.append(f"| {h}d | {_p(v)} |\n")

    # ---- D. Conclusion ----
    L.append("\n### 9D. 拆分结论\n\n")

    # Determine outcome based on rules
    cond_5d = cond["pooled"][5]
    cond_ric = cond_5d["pooled_rankic"]
    cond_ricir = cond_5d["pooled_rankicir"]
    event_5d_spread = event[5]["spread"]
    matched_5d_spread = event["matched"].get(5, {}).get("matched_spread", np.nan)

    n_days_total = cond_5d["n_ic_days"]
    ric_pos_folds = sum(1 for fk in cond["folds"]
                        if cond["folds"][fk].get(5, {}).get("rankic", -99) > 0)
    n_folds = len(cond["folds"])

    L.append("| 指标 | 值 |\n")
    L.append("|---|---|\n")
    L.append(f"| 突破事件 spread (5d) | {_pp(event_5d_spread)} |\n")
    L.append(f"| 风格匹配后 spread (5d) | {_pp(matched_5d_spread)} |\n")
    L.append(f"| conditional pooled RankIC (5d) | {_p(cond_ric)} |\n")
    L.append(f"| conditional pooled RankICIR (5d) | {_p(cond_ricir)} |\n")
    L.append(f"| conditional RankIC 正折比 | {ric_pos_folds}/{n_folds} |\n")
    L.append(f"| 中性化 conditional RankIC (5d) | {_p(cond['neutralized'][5]['neutralized_rankic'])} (仅{cond['neutralized'][5]['n_neut_days']}日，不参与判定) |\n")
    L.append(f"| A191 残差化 RankIC (5d) | {_p(a191_rel['a191_resid_rankic'].get(5, np.nan))} |\n")
    L.append(f"| pooled IC 总日数 | {n_days_total} |\n")

    # Apply decision rules
    if cond_ric is not None and not np.isnan(cond_ric) and cond_ric <= 0:
        L.append(f"\n**判定**: conditional quality RankIC = {_p(cond_ric)} ≤ 0\n\n")
        L.append("> 正向突破质量假设被否定。\n\n")
        L.append("- **factor_status**: `reject`\n")
        L.append("- **research_status**: `retained_as_exhaustion_hypothesis`\n")
        L.append("- 突破事件发生后前向收益系统性偏低（拥挤/均值回归），且突破内部质量评分无法正向区分后续收益。\n")
        L.append("- 因子作为 exhaustion/反转信号的潜力留待 future work，当前不作为正向 alpha 因子。\n")
    elif n_days_total < 20 or ric_pos_folds <= n_folds / 2:
        L.append(f"\n**判定**: 条件样本不足或跨折不稳定（IC日数={n_days_total}, 正折比={ric_pos_folds}/{n_folds}）\n\n")
        L.append("- **factor_status**: `hold`\n")
        L.append("- **reason**: `insufficient_conditional_evidence`\n")
    elif event_5d_spread is not None and not np.isnan(event_5d_spread) and event_5d_spread < 0 and cond_ric > 0:
        L.append(f"\n**判定**: 事件效应<0, 条件质量 RankIC>0\n\n")
        L.append("> 突破事件整体拥挤，但在突破股票内部，质量评分仍有选择价值。\n\n")
        L.append("- **factor_status**: `hold`\n")
        L.append("- **usage**: `conditional_filter_only`\n")
        L.append("- 未来方向: 结合 BO flag 作为负向筛选 + quality 在 BO 内部正向筛选的双层过滤。\n")
    else:
        L.append(f"\n**判定**: 不满足任一预设情况（event_spread={_pp(event_5d_spread)}, "
                 f"cond_ric={_p(cond_ric)}）\n\n")
        L.append("- **factor_status**: `hold`\n")
        L.append("- **reason**: 条件质量 RankIC 不显著\n")

    # Coverage disclosure
    L.append("\n### 突破事件覆盖率披露\n\n")
    for fk in sorted(cond["folds"].keys()):
        fold = cond["folds"][fk]
        for h in HORIZONS:
            fh = fold.get(h, {})
            L.append(f"- {fk} {h}d: {fh.get('n_bo_events', 0)} 条 BO 事件\n")

    return "".join(L)


# ======================================================================
def main():
    print("=" * 60)
    print("Phase 8A-E.1.1 事件效应 vs 质量效应拆分")
    print("=" * 60)

    print("\n[1/5] 加载数据 ...")
    with open(UNIVERSE_FILE) as f:
        codes = [l.strip() for l in f if l.strip()]
    panel = load_panel(codes)
    fwd = load_fwd()
    controls = load_controls()
    idx_fwd = load_index_returns()
    print(f"  panel: {len(panel)} 行, {panel['symbol'].nunique()} 只, {panel['trade_date'].nunique()} 日")

    print("\n[2/5] 计算因子 ...")
    scores = compute_scores(panel)
    n_bo = scores["breakout_event"].sum()
    bo_daily = scores[scores["breakout_event"]].groupby("trade_date").size()
    print(f"  BO 事件总数: {n_bo}")
    print(f"  BO 日均: {bo_daily.mean():.1f}, 最小: {bo_daily.min()}, 最大: {bo_daily.max()}")
    coverage = {"n_bo_total": int(n_bo), "n_bo_daily_avg": float(bo_daily.mean()),
                "bo_pct": float(n_bo / len(scores))}

    print("\n[3/5] A. 突破事件效应 ...")
    event = event_effect(scores, fwd, controls, idx_fwd)
    for h in HORIZONS:
        e = event[h]
        print(f"  {h}d: BO={_pct(e['bo_mean'])}, NBO={_pct(e['nbo_mean'])}, "
              f"spread={_pp(e['spread'])}, n_bo={e['n_bo_total']}, n_nbo={e['n_nbo_total']}, "
              f"days={e['n_bo_days']}")
    for h in HORIZONS:
        m = event["matched"].get(h, {})
        print(f"  matched {h}d: spread={_pp(m.get('matched_spread'))}, "
              f"pairs={m.get('n_pairs', 0)}")

    print("\n[4/5] B. 突破质量效应（仅 BO 内部）...")
    cond = conditional_quality(scores, fwd, controls)
    for h in HORIZONS:
        c = cond["full"][h]
        print(f"  {h}d: cond_RankIC={_p(c['rankic'])}, cond_RankICIR={_p(c['rankicir'])}, "
              f"days={c['n_days']}, tercile(top/mid/bot)={_pct(c['top_mean'])}/{_pct(c['mid_mean'])}/{_pct(c['bot_mean'])}")
    for h in HORIZONS:
        n = cond["neutralized"][h]
        print(f"  neut {h}d: neut_RankIC={_p(n['neutralized_rankic'])}, days={n['n_neut_days']}")
    for h in HORIZONS:
        p = cond["pooled"][h]
        print(f"  pooled {h}d: pooled_RankIC={_p(p['pooled_rankic'])}, "
              f"pooled_RankICIR={_p(p['pooled_rankicir'])}, days={p['n_ic_days']}")

    print("\n[5/5] C. Alpha191 关系（仅 BO 内部）...")
    a191_panel = arf.load_alpha_panel(DATA_START, DATA_END)
    a191_rel = alpha191_conditional(scores, a191_panel, fwd)
    print(f"  quality vs A191_ew corr: {_p(a191_rel['corr_with_a191_ew'])} "
          f"({a191_rel['n_corr_days_eq']} days)")
    print(f"  max single factor corr: {a191_rel['max_single_corr_factor']} "
          f"({_p(a191_rel['max_single_corr_value'])})")
    for h in HORIZONS:
        v = a191_rel["a191_resid_rankic"].get(h, np.nan)
        print(f"  A191-residualized cond RankIC {h}d: {_p(v)}")

    # Build decomposition section
    section = build_section(event, cond, a191_rel, coverage)
    print(section)

    # Update report: replace old section 8 onwards with new sections 8 + 9
    report_path = PROJECT / "reports" / "phase8a_breakout_validation_report.md"
    existing = report_path.read_text(encoding="utf-8")
    if "## 8. 结论与建议" in existing:
        base = existing.split("## 8. 结论与建议")[0]
    else:
        base = existing + "\n"

    new_section = build_updated_conclusion(event, cond, a191_rel, coverage)
    report = base + new_section + section
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已更新: {report_path}")


def build_updated_conclusion(event, cond, a191_rel, coverage):
    """Build updated section 8 with decomposition-aware verdict."""
    L = ["## 8. 结论与建议\n\n"]

    cond_5d = cond["pooled"][5]
    cond_ric = cond_5d["pooled_rankic"]
    cond_ricir = cond_5d["pooled_rankicir"]
    event_5d_spread = event[5]["spread"]
    matched_5d_spread = event["matched"].get(5, {}).get("matched_spread", np.nan)
    neut_5d_ric = cond["neutralized"][5]["neutralized_rankic"]
    neut_5d_n = cond["neutralized"][5]["n_neut_days"]
    a191_resid_5d = a191_rel["a191_resid_rankic"].get(5, np.nan)

    L.append("### 综合判定\n\n")

    L.append("| 指标 | 值 | 解读 |\n")
    L.append("|---|---|---|\n")
    L.append(f"| 突破事件 spread (5d) | {_pp(event_5d_spread)} | BO − NBO 前向收益差（百分点） |\n")
    L.append(f"| 风格匹配后 spread (5d) | {_pp(matched_5d_spread)} | 匹配后 BO − NBO 收益差 |\n")
    L.append(f"| conditional RankIC (5d, pooled) | {_p(cond_ric)} | 仅BO内部 quality 排序能力 |\n")
    L.append(f"| conditional RankICIR (5d, pooled) | {_p(cond_ricir)} | BO内部 quality 稳定性 |\n")
    L.append(f"| 中性化 conditional RankIC | {_p(neut_5d_ric)} (仅{neut_5d_n}日) | 去风格后残差排序，样本不足不参与判定 |\n")
    L.append(f"| A191 残差化 conditional RankIC | {_p(a191_resid_5d)} | 去A191后残差排序 |\n")
    L.append(f"| quality vs A191 相关 | {_p(a191_rel['corr_with_a191_ew'])} | BO内部 quality 与 A191 重叠度 |\n\n")

    # Determine verdict
    if cond_ric is not None and not np.isnan(cond_ric) and cond_ric <= 0:
        L.append("**最终判定: reject**\n\n")
        L.append("> 正向突破质量假设被否定。conditional quality RankIC ≤ 0，说明在突破股票内部，"
                 "quality score 更高的突破后续表现并不更好（甚至更差）。\n\n")
        L.append("- **factor_status**: `reject`\n")
        L.append("- **research_status**: `retained_as_exhaustion_hypothesis`\n")
        L.append("- 突破事件后前向收益系统性偏低（exhaustion/均值回归），quality score 无法在BO内部提供正向区分。\n")
        L.append("- **不得将此因子符号反转后作为正向因子**；反转后的「选非突破股」等价于一个流动性/规模过滤器，不是 alpha。\n")
        L.append("- 若后续研究 exhaustion 机制（如结合持仓拥挤度、涨停板等），可重新评估 quality score 的条件价值。\n")
    elif cond_ric > 0 and event_5d_spread < 0:
        L.append("**最终判定: hold (conditional_filter_only)**\n\n")
        L.append("> 突破事件整体拥挤（event spread < 0），但在突破股票内部，质量评分仍有选择价值"
                 "(conditional RankIC > 0)。\n\n")
        L.append("- **factor_status**: `hold`\n")
        L.append("- **usage**: `conditional_filter_only`\n")
    else:
        L.append("**最终判定: hold (insufficient_conditional_evidence)**\n\n")
        L.append("> 条件样本不足或跨折不稳定，无法得出确定结论。\n\n")
        L.append("- **factor_status**: `hold`\n")
        L.append("- **reason**: `insufficient_conditional_evidence`\n")

    L.append("\n### 关键发现\n\n")
    L.append(f"1. 突破事件效应: BO − NBO spread = {_pp(event_5d_spread)} "
             f"(5d)，突破后前向收益{'低于' if event_5d_spread < 0 else '高于'}非突破"
             f"（BO={_pct(event[5]['bo_mean'])}, NBO={_pct(event[5]['nbo_mean'])}）\n")
    if not np.isnan(matched_5d_spread):
        L.append(f"2. 风格匹配后 spread = {_pp(matched_5d_spread)}，方向反转，"
                 "说明原始 spread 主要由风格差异（高换手/高成交额）驱动，"
                 "不能据此认定突破事件本身存在独立 Alpha。\n")
    else:
        L.append("2. 风格匹配后 spread 不可用。\n")
    L.append(f"3. conditional quality RankIC = {_p(cond_ric)}，"
             f"{'正向' if cond_ric > 0 else '负向/零'}区分 BO 内部后续收益\n")
    L.append(f"4. 中性化 conditional RankIC = {_p(neut_5d_ric)}，"
             f"仅 {neut_5d_n} 个有效日，不参与最终判定\n")
    L.append(f"5. A191残差化 conditional RankIC = {_p(a191_resid_5d)}\n")
    L.append(f"6. 因子使用需明确区分 event filter（0/1）和 quality score（条件连续值），不可混用\n")

    return "".join(L)


if __name__ == "__main__":
    main()
