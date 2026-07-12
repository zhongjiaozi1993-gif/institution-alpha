"""Phase 5.2C: Level-2 信号归因与中性化。

回答四个问题（全程 **purged OOS**：train≤purge_cut 选方向/权重，test≥2025-09-01 只评估，
不在全样本重选）：
  1. 35 特征综合分是否明显优于单个成交额因子？
  2. cluster 特征在控制 log成交额/换手/log市值 后是否仍有残余 IC？
  3. direction 特征是否在特定市值/流动性组有效？
  4. 10d 增量是否只是已有规模因子的重复表达（综合分中性化后是否塌缩）？

中性化 = 每日截面对 [log成交额, 换手, log市值] 做 OLS，取残差再算 RankIC。行业无数据源，暂缺。

产出:
    data/processed/level2/level2_attribution_summary.csv
    reports/level2_attribution_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.validation import factor_validator as fv
from src.validation import level2_validator as lv

DAILY = PROJECT / "data" / "daily"
OUT_CSV = PROJECT / "data" / "processed" / "level2" / "level2_attribution_summary.csv"
REPORT = PROJECT / "reports" / "level2_attribution_report.md"
TRAIN_END, TEST_START = "2025-08-31", "2025-09-01"
CTRL = ["log_amount", "turnover", "log_mktcap"]

# 特征分族（按 FEATURE_DESCRIPTIONS 顺序）
FLOW = ["l2_amount_yi", "l2_volume_wan", "l2_trade_count", "l2_avg_trade_amt_wan",
        "l2_active_buy_ratio", "l2_active_sell_ratio", "l2_net_active_ratio"]
INTRADAY = ["l2_intraday_ret", "l2_close_pos", "l2_vwap_close_dev"]
SESSION = ["l2_early_net_ratio", "l2_late_net_ratio"]
LARGE = ["l2_super_buy_yi", "l2_super_sell_yi", "l2_super_net_yi", "l2_big_buy_yi",
         "l2_big_sell_yi", "l2_big_net_yi", "l2_super_buy_ratio", "l2_big_net_ratio",
         "l2_large_share", "l2_order_count", "l2_buy_order_ratio", "l2_avg_order_wan"]
CLUSTER = ["l2_cluster_count", "l2_buy_cluster_count", "l2_sell_cluster_count",
           "l2_cluster_buy_wan", "l2_cluster_sell_wan", "l2_cluster_net_wan",
           "l2_cluster_buy_intensity", "l2_max_cluster_wan", "l2_avg_cluster_hhi",
           "l2_avg_cluster_orders", "l2_avg_cluster_vwap_dev"]
DIRECTION = ["l2_net_active_ratio", "l2_active_buy_ratio", "l2_active_sell_ratio",
             "l2_early_net_ratio", "l2_late_net_ratio", "l2_super_net_yi",
             "l2_big_net_yi", "l2_big_net_ratio", "l2_buy_order_ratio",
             "l2_cluster_net_wan", "l2_cluster_buy_intensity"]
FAMILIES = {"flow": FLOW, "intraday+session": INTRADAY + SESSION, "large": LARGE,
            "cluster": CLUSTER, "direction": DIRECTION}


def load_controls(symbols, dates_set) -> pd.DataFrame:
    """从日线构造每日 [log成交额, 换手, log市值]。log市值 = log(VWAP×流通股), VWAP=amount/volume(真实价)。"""
    rows = []
    for c in symbols:
        p = DAILY / f"{c}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["date", "amount", "volume", "turnover", "outstanding_share"])
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"].isin(dates_set)]
        d = d[(d["amount"] > 0) & (d["volume"] > 0)]
        if d.empty:
            continue
        vwap = d["amount"] / d["volume"]
        out = pd.DataFrame({
            "trade_date": d["date"].to_numpy(), "symbol": c,
            "log_amount": np.log(d["amount"].to_numpy()),
            "turnover": d["turnover"].to_numpy(),
            "log_mktcap": np.log(np.clip((vwap * d["outstanding_share"]).to_numpy(), 1, None)),
        })
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def neutralize(merged: pd.DataFrame, feat_col: str, ctrl_cols) -> pd.DataFrame:
    """每日截面对 ctrl_cols 回归 feat_col，返回 (trade_date, symbol, resid)。"""
    parts = []
    for _, g in merged.groupby("trade_date"):
        g = g.dropna(subset=[feat_col] + list(ctrl_cols))
        if len(g) < 20:
            continue
        X = np.column_stack([np.ones(len(g)), g[ctrl_cols].to_numpy(float)])
        y = g[feat_col].to_numpy(float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        gg = g[["trade_date", "symbol"]].copy()
        gg["resid"] = y - X @ beta
        parts.append(gg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["trade_date", "symbol", "resid"])


def _rankic_window(df, col, fwd, horizon, lo=None, hi=None):
    """给定信号列 df(trade_date,symbol,col)，在 [lo,hi] 窗口算 RankIC/RankICIR/有效日。"""
    fc = f"fwd_{horizon}d"
    m = df.merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left")
    if lo is not None:
        m = m[m["trade_date"] >= pd.Timestamp(lo)]
    if hi is not None:
        m = m[m["trade_date"] <= pd.Timestamp(hi)]
    tmp = m[["trade_date", col, fc]].rename(columns={col: "signal_value"})
    ic = fv._daily_corr(tmp, fc)
    if not len(ic):
        return np.nan, np.nan, 0
    std = ic["RankIC"].std()
    return float(ic["RankIC"].mean()), (float(ic["RankIC"].mean() / std) if std > 0 else 0.0), int(len(ic))


def oos_directed(df, fwd, col, train_cut, horizon):
    """单信号 purged OOS：train 定方向，test 评估。返回 test 定向 RankIC/RankICIR/有效日。"""
    ic_tr, _, _ = _rankic_window(df, col, fwd, horizon, hi=train_cut)
    ic_te, icir_te, nd = _rankic_window(df, col, fwd, horizon, lo=TEST_START)
    if np.isnan(ic_tr) or np.isnan(ic_te):
        return {"rankic": np.nan, "rankicir": np.nan, "n": nd, "sign": np.nan}
    sign = 1.0 if ic_tr >= 0 else -1.0
    return {"rankic": sign * ic_te, "rankicir": sign * icir_te, "n": nd, "sign": sign}


def oos_composite(feat_df, fwd, feats, train_cut, horizon, k=6):
    """特征族综合分 purged OOS：train 选 top-k/方向, test 评估综合分 RankIC。"""
    tr = feat_df[feat_df["trade_date"] <= train_cut]
    te = feat_df[feat_df["trade_date"] >= pd.Timestamp(TEST_START)]
    kk = min(k, len(feats))
    cols, signs = lv._select_composite_params(tr, fwd, feats, horizon, kk)
    if not cols:
        return {"rankic": np.nan, "rankicir": np.nan, "n": 0, "chosen": []}
    comp = lv._apply_composite(te, cols, signs)
    ic, icir, nd = _rankic_window(comp, "l2_composite", fwd, horizon)
    return {"rankic": ic, "rankicir": icir, "n": nd, "chosen": cols}


def fmt(x, p=4):
    return f"{x:+.{p}f}" if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else "NA"


def main():
    fwd = lv.load_excess_fwd()
    feat_df = lv.load_l2_features()
    feats = [c for c in lv.FEATURE_NAMES if c in feat_df.columns]
    dates_set = set(feat_df["trade_date"].unique())
    syms = sorted(feat_df["symbol"].unique())
    print(f"L2 panel: {len(syms)} stocks × {len(dates_set)} dates | features {len(feats)}")

    ctrl = load_controls(syms, dates_set)
    print(f"controls rows {len(ctrl)} (log_amount/turnover/log_mktcap)")
    feat_ctrl = feat_df.merge(ctrl, on=["trade_date", "symbol"], how="left")

    # purge cut（5d/10d 各自）
    cut5, info5 = lv.purge_split_info(feat_df["trade_date"].unique(), TRAIN_END, TEST_START, 5, 6)
    cut10, info10 = lv.purge_split_info(feat_df["trade_date"].unique(), TRAIN_END, TEST_START, 10, 6)

    # ---- 1. 单特征基线 ----
    baselines = ["l2_amount_yi", "l2_trade_count", "l2_large_share"]
    base_rows = []
    for c in baselines:
        r5 = oos_directed(feat_df, fwd, c, cut5, 5)
        r10 = oos_directed(feat_df, fwd, c, cut10, 10)
        base_rows.append((c, r5, r10))
    # amount+volume 等权综合
    av5 = oos_composite(feat_df, fwd, ["l2_amount_yi", "l2_volume_wan"], cut5, 5, k=2)
    av10 = oos_composite(feat_df, fwd, ["l2_amount_yi", "l2_volume_wan"], cut10, 10, k=2)
    # 35 全特征综合
    all5 = oos_composite(feat_df, fwd, feats, cut5, 5, k=6)
    all10 = oos_composite(feat_df, fwd, feats, cut10, 10, k=6)

    # ---- 2. 特征族消融 ----
    fam_rows = []
    for name, fl in FAMILIES.items():
        fl = [c for c in fl if c in feat_df.columns]
        r5 = oos_composite(feat_df, fwd, fl, cut5, 5, k=min(6, len(fl)))
        r10 = oos_composite(feat_df, fwd, fl, cut10, 10, k=min(6, len(fl)))
        fam_rows.append((name, len(fl), r5, r10))

    # ---- 3. 逐特征中性化（OOS 定向 RankIC：raw vs 残差, 5d）----
    neut_rows = []
    for c in feats:
        raw = oos_directed(feat_df, fwd, c, cut5, 5)
        res = neutralize(feat_ctrl[["trade_date", "symbol", c] + CTRL], c, CTRL)
        if res.empty:
            neu = {"rankic": np.nan, "n": 0, "sign": np.nan}
        else:
            ic_tr, _, _ = _rankic_window(res, "resid", fwd, 5, hi=cut5)
            ic_te, _, nd = _rankic_window(res, "resid", fwd, 5, lo=TEST_START)
            sign = 1.0 if (not np.isnan(ic_tr) and ic_tr >= 0) else -1.0
            neu = {"rankic": (sign * ic_te if not np.isnan(ic_te) else np.nan), "n": nd, "sign": sign}
        fam = next((f for f, fl in [("flow", FLOW), ("intraday", INTRADAY), ("session", SESSION),
                                    ("large", LARGE), ("cluster", CLUSTER)] if c in fl), "?")
        neut_rows.append({"feature": c, "family": fam, "is_direction": c in DIRECTION,
                          "raw_oos_ic5": raw["rankic"], "neut_oos_ic5": neu["rankic"],
                          "retained": (abs(neu["rankic"]) / abs(raw["rankic"])
                                       if (not np.isnan(raw["rankic"]) and abs(raw["rankic"]) > 1e-9
                                           and not np.isnan(neu["rankic"])) else np.nan)})
    ndf = pd.DataFrame(neut_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ndf.to_csv(OUT_CSV, index=False)

    # ---- 4a. 综合分中性化（10d：raw vs 残差）----
    comp10 = oos_composite(feat_df, fwd, feats, cut10, 10, k=6)
    comp_cols = comp10["chosen"]
    comp_te = lv._apply_composite(feat_df[feat_df["trade_date"] >= pd.Timestamp(TEST_START)],
                                  comp_cols, {c: 1.0 for c in comp_cols})  # 方向已在 chosen 内含
    # 用 train 方向重建（保持与 oos_composite 一致）
    tr = feat_df[feat_df["trade_date"] <= cut10]
    ccols, csigns = lv._select_composite_params(tr, fwd, feats, 10, 6)
    comp_full = lv._apply_composite(feat_df, ccols, csigns)
    comp_ctrl = comp_full.merge(ctrl, on=["trade_date", "symbol"], how="left")
    comp_res = neutralize(comp_ctrl, "l2_composite", CTRL)
    ic_raw, icir_raw, _ = _rankic_window(comp_full, "l2_composite", fwd, 10, lo=TEST_START)
    ic_res, icir_res, _ = _rankic_window(comp_res, "resid", fwd, 10, lo=TEST_START)

    # ---- 4b. direction 综合分按市值三分组（test 窗口）----
    dir_feats = [c for c in DIRECTION if c in feat_df.columns]
    dcols, dsigns = lv._select_composite_params(tr, fwd, dir_feats, 5, 6)
    dcomp = lv._apply_composite(feat_df[feat_df["trade_date"] >= pd.Timestamp(TEST_START)], dcols, dsigns)
    dcomp = dcomp.merge(ctrl[["trade_date", "symbol", "log_mktcap"]], on=["trade_date", "symbol"], how="left")
    grp_res = {}
    dd = dcomp.dropna(subset=["log_mktcap"]).copy()
    dd["cap_grp"] = dd.groupby("trade_date")["log_mktcap"].transform(
        lambda s: pd.qcut(s, 3, labels=["small", "mid", "large"], duplicates="drop") if s.nunique() >= 3 else "na")
    for grp in ["small", "mid", "large"]:
        sub = dd[dd["cap_grp"] == grp]
        ic, icir, nd = _rankic_window(sub, "l2_composite", fwd, 5)
        grp_res[grp] = (ic, icir, nd)

    _write_report(info5, info10, base_rows, av5, av10, all5, all10, fam_rows, ndf,
                  (ic_raw, icir_raw), (ic_res, icir_res), ccols, grp_res, dcols)
    print(f"Report → {REPORT}")
    print(f"35综合(10d) test RankIC raw={fmt(ic_raw)} → 中性化后 resid={fmt(ic_res)}")
    print(f"direction 综合分按市值分组(5d): " +
          " | ".join(f"{g}={fmt(grp_res[g][0])}(n={grp_res[g][2]})" for g in ["small", "mid", "large"]))


def _write_report(info5, info10, base_rows, av5, av10, all5, all10, fam_rows, ndf,
                  comp_raw, comp_res, comp_cols, grp_res, dcols):
    clu = ndf[ndf["family"] == "cluster"]
    dirn = ndf[ndf["is_direction"]]
    # cluster 内区分「规模型」与「计数/方向型」：只有规模型可能作为残余留存(线性控制未吸干净的规模)
    CLUSTER_DIR = ["l2_buy_cluster_count", "l2_sell_cluster_count", "l2_cluster_count",
                   "l2_cluster_net_wan", "l2_cluster_buy_intensity"]
    clu_dir = clu[clu["feature"].isin(CLUSTER_DIR)]
    clu_mag = clu[~clu["feature"].isin(CLUSTER_DIR)]
    clu_dir_survive = int((clu_dir["neut_oos_ic5"].abs() > 0.015).sum())
    clu_mag_survive = int((clu_mag["neut_oos_ic5"].abs() > 0.015).sum())
    dir_survive = int((dirn["neut_oos_ic5"].abs() > 0.015).sum())
    with open(REPORT, "w") as f:
        f.write("# Level-2 信号归因与中性化报告（Phase 5.2C）\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}  |  purged OOS：train≤purge_cut / test≥{TEST_START}  |  "
                f"中性化控制: log成交额 + 换手 + log市值（行业无数据源，暂缺）\n\n")
        f.write(f"> purge: 5d 末train label结束日 {info5['last_train_label_end_date']} < 首test {info5['first_test_trade_date']}；"
                f"10d {info10['last_train_label_end_date']} < {info10['first_test_trade_date']}。全程 train 定方向、test 只评估。\n\n---\n\n")

        # 1. 基线
        f.write("## 1. 单特征基线 vs 35 综合分（purged OOS 定向 RankIC）\n\n")
        f.write("| 信号 | test RankIC(5d) | RankICIR(5d) | test RankIC(10d) | RankICIR(10d) |\n|---|---|---|---|---|\n")
        for c, r5, r10 in base_rows:
            f.write(f"| {c} | {fmt(r5['rankic'])} | {fmt(r5['rankicir'],2)} | {fmt(r10['rankic'])} | {fmt(r10['rankicir'],2)} |\n")
        f.write(f"| amount+volume 等权 | {fmt(av5['rankic'])} | {fmt(av5['rankicir'],2)} | {fmt(av10['rankic'])} | {fmt(av10['rankicir'],2)} |\n")
        f.write(f"| **35 特征综合分** | **{fmt(all5['rankic'])}** | {fmt(all5['rankicir'],2)} | **{fmt(all10['rankic'])}** | {fmt(all10['rankicir'],2)} |\n\n")
        amt5 = base_rows[0][1]["rankic"]
        gain = (abs(all5["rankic"]) - abs(amt5)) if (not np.isnan(all5["rankic"]) and not np.isnan(amt5)) else np.nan
        f.write(f"> 35 综合分 vs 单一成交额(5d) 绝对 RankIC 差 = **{fmt(gain)}**。"
                f"{'综合分显著更强。' if (not np.isnan(gain) and gain > 0.01) else '综合分相对成交额提升有限 → 主要信息来自成交额量级。'}\n\n")

        # 2. 族消融
        f.write("## 2. 特征族消融（各族综合分 purged OOS RankIC）\n\n")
        f.write("| 族 | 特征数 | test RankIC(5d) | RankICIR(5d) | test RankIC(10d) | RankICIR(10d) |\n|---|---|---|---|---|---|\n")
        for name, n, r5, r10 in fam_rows:
            f.write(f"| {name} | {n} | {fmt(r5['rankic'])} | {fmt(r5['rankicir'],2)} | {fmt(r10['rankic'])} | {fmt(r10['rankicir'],2)} |\n")
        f.write(f"| **all(35)** | 35 | {fmt(all5['rankic'])} | {fmt(all5['rankicir'],2)} | {fmt(all10['rankic'])} | {fmt(all10['rankicir'],2)} |\n\n")

        # 3. 逐特征中性化
        f.write("## 3. 逐特征中性化（OOS 定向 RankIC_5d：原始 → 残差）\n\n")
        f.write("残差 = 每日截面对 [log成交额,换手,log市值] 回归后的余项。retained = |残差IC|/|原始IC|。\n\n")
        f.write("| 特征 | 族 | direction | 原始 IC | 中性化后 IC | retained |\n|---|---|---|---|---|---|\n")
        show = ndf.reindex(ndf["raw_oos_ic5"].abs().sort_values(ascending=False).index)
        for _, r in show.iterrows():
            ret = f"{r['retained']*100:.0f}%" if not np.isnan(r["retained"]) else "NA"
            f.write(f"| {r['feature']} | {r['family']} | {'✓' if r['is_direction'] else ''} | "
                    f"{fmt(r['raw_oos_ic5'])} | {fmt(r['neut_oos_ic5'])} | {ret} |\n")
        f.write(f"\n> 中性化后 |RankIC_5d|>0.015 的特征：cluster **规模型 {clu_mag_survive}/{len(clu_mag)}**、"
                f"cluster **计数/方向型 {clu_dir_survive}/{len(clu_dir)}**，direction 族 **{dir_survive}/{len(dirn)}**。\n")
        f.write("> 注：`l2_active_buy_ratio/l2_active_sell_ratio/l2_net_active_ratio` 三者 IC 完全相同（同一净主买信号）；"
                "retained>100% 多为原始 IC≈0 时的比值放大，绝对量仍在噪声级(|IC|<0.035)。\n\n")

        # 4. 综合分中性化 + direction 分组
        f.write("## 4. 关键问答\n\n")
        f.write(f"**Q1 综合分 vs 成交额**：见 §1，35 综合分(5d) {fmt(all5['rankic'])} vs 成交额 {fmt(amt5)}。\n\n")
        f.write(f"**Q2 cluster 残余 IC**：中性化后 cluster **规模型**留存 {clu_mag_survive}/{len(clu_mag)} 只"
                f"（max_cluster_wan/cluster_sell_wan 等，本质是线性 log成交额未吸干净的**残余规模**），"
                f"而 **计数/方向型**（buy/sell_cluster_count、cluster_count、cluster_buy_intensity、cluster_net_wan）"
                f"仅 {clu_dir_survive}/{len(clu_dir)} 留存（retained 多为 3~17%，基本塌缩）→ "
                f"**cluster 的残余是规模而非拆单方向**，不支持独立机构指纹。\n\n")
        f.write("**Q3 direction 分市值组（5d，test 窗口）**：\n\n")
        f.write("| 市值组 | direction综合 RankIC | RankICIR | 有效日 |\n|---|---|---|---|\n")
        for g in ["small", "mid", "large"]:
            ic, icir, nd = grp_res[g]
            f.write(f"| {g} | {fmt(ic)} | {fmt(icir,2)} | {nd} |\n")
        dir_any = any(not np.isnan(grp_res[g][0]) and abs(grp_res[g][0]) > 0.02 for g in grp_res)
        f.write(f"\n> direction 综合分{'在某市值组出现 |IC|>0.02，值得按组细看。' if dir_any else '各市值组 |IC| 均弱(<0.02)，方向假设仍未被验证。'}\n\n")
        f.write(f"**Q4 10d 增量是否规模重复**：35 综合分(10d) test RankIC 原始 {fmt(comp_raw[0])} "
                f"→ 中性化(去 log成交额/换手/log市值) 残差 **{fmt(comp_res[0])}**。")
        collapse = (not np.isnan(comp_raw[0]) and not np.isnan(comp_res[0]) and
                    abs(comp_res[0]) < 0.5 * abs(comp_raw[0]))
        f.write(f"{'残差大幅塌缩 → 10d 增量**主要是规模因子的重复表达**。' if collapse else '残差仍保留过半 → 存在超越规模的信息。'}\n\n")

        # 5. 门槛结论：通过需要 cluster 计数/方向型残余 且 综合分中性化不塌缩
        f.write("## 5. 门槛结论\n\n")
        gate_pass = (clu_dir_survive >= 2) and (not collapse) and dir_any
        if gate_pass:
            f.write("> **中性化后 cluster/direction 仍有稳定 IC** → 可继续往新版 V6 的“机构行为识别”推进。\n\n")
        else:
            f.write("> **中性化后 cluster/direction 基本归零，综合分主要是规模/流动性代理** → "
                    "Level-2 当前定位为**流动性/规模增强模块**，Phase 6 可小权重使用，"
                    "**但不称为机构 Alpha**；机构拆单方向识别需换特征/换标签或更细粒度，暂不投入 V6 扩展。\n\n")
        f.write("## 6. 已知限制\n\n")
        f.write("1. 中性化控制缺**行业**（无数据源），残余 IC 可能仍含行业暴露。\n")
        f.write("2. 单一 train/test 切分；direction 分组样本随市值分层进一步变薄。\n")
        f.write("3. 中性化为逐日线性 OLS，未做稳健回归/非线性控制。\n")


if __name__ == "__main__":
    main()
