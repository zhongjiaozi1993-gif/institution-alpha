"""在 Universe_A/B/C 上重跑 30 个 Alpha191 因子，输出稳定性报告。

用法:
    python3 scripts/run_alpha191_validation.py

产出:
    data/processed/signals/alpha191/{Universe}_summary.csv
    reports/alpha191_factor_detail/{signal_id}.md
    reports/alpha191_validation_summary.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.validation import factor_validator as fv
from src.registry import universe_registry as reg

START, END = "2025-01-01", "2025-12-31"
SIGNAL_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191_full"
OUT_DIR = PROJECT / "data" / "processed" / "signals" / "alpha191"
DETAIL_DIR = PROJECT / "reports" / "alpha191_factor_detail"
SUMMARY = PROJECT / "reports" / "alpha191_validation_summary.md"
UNIVERSES = ["Universe_A", "Universe_B", "Universe_C"]


def fmt(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else "NA"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    fwd = fv.load_labels_as_fwd()
    sig_files = sorted(SIGNAL_DIR.glob("signal*.parquet"))
    print(f"Validating {len(sig_files)} factors × {len(UNIVERSES)} universes")

    uni_syms = {u: set(reg.load_universe(u)) for u in UNIVERSES}
    uni_meta = {u: reg.load_universe_table(u) for u in UNIVERSES}
    for u in UNIVERSES:
        uni_meta[u]["symbol"] = uni_meta[u]["symbol"].astype(str).str.zfill(6)

    # results[universe] = list of row dicts
    results = {u: [] for u in UNIVERSES}
    for sp in sig_files:
        for u in UNIVERSES:
            r = fv.validate_factor(sp, fwd, uni_syms[u], uni_meta[u], START, END)
            r["universe"] = u
            results[u].append(r)
        print(f"  {sp.stem} done")

    # ---- per-universe summary CSV (flatten dict cols out) ----
    drop_cols = ["quarterly_rankic_5d", "monthly_rankic_5d", "mktcap_rankic_5d", "liq_rankic_5d"]
    for u in UNIVERSES:
        df = pd.DataFrame(results[u]).drop(columns=drop_cols, errors="ignore")
        df.to_csv(OUT_DIR / f"{u}_summary.csv", index=False)
    print(f"Saved per-universe summaries to {OUT_DIR}")

    # index by signal_id for cross-universe view
    by_sig = {}
    for u in UNIVERSES:
        for r in results[u]:
            by_sig.setdefault(r["signal_id"], {})[u] = r

    # ---- per-factor detail reports ----
    for sid, per_u in sorted(by_sig.items()):
        rB = per_u.get("Universe_B", {})
        sname = rB.get("signal_name", "")
        with open(DETAIL_DIR / f"{sid.lower()}.md", "w") as f:
            f.write(f"# {sid}: {sname} — 跨 Universe 稳定性\n\n")
            f.write(f"窗口: {START} ~ {END}  |  label: open-to-open(无未来函数)\n\n")
            f.write("## RankIC / RankICIR / spread（各 horizon × universe）\n\n")
            f.write("| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for u in UNIVERSES:
                r = per_u.get(u, {})
                f.write(f"| {u} | {r.get('n_stocks','')} | {fmt(r.get('RankIC_1d'))} | "
                        f"{fmt(r.get('RankIC_5d'))} | {fmt(r.get('RankIC_10d'))} | "
                        f"{fmt(r.get('RankICIR_5d'),2)} | {fmt(r.get('spread_5d'),3)} | "
                        f"{fmt(r.get('cost_adj_spread_5d'),3)} | {fmt(r.get('coverage_5d'),1)} |\n")
            f.write("\n## 分季度 RankIC_5d（替代分年度）\n\n")
            for u in UNIVERSES:
                q = per_u.get(u, {}).get("quarterly_rankic_5d", {})
                f.write(f"- {u}: " + (", ".join(f"Q{k}={v:+.4f}" for k, v in q.items()) if q else "NA") + "\n")
            f.write("\n## 分市值组 / 分流动性组 RankIC_5d（Universe_B）\n\n")
            mk = rB.get("mktcap_rankic_5d", {})
            lq = rB.get("liq_rankic_5d", {})
            f.write("- 市值组: " + (", ".join(f"{k}={v:+.4f}" for k, v in mk.items()) if mk else "NA") + "\n")
            f.write("- 流动性组: " + (", ".join(f"{k}={v:+.4f}" for k, v in lq.items()) if lq else "NA") + "\n")
            f.write(f"- 换手率(5d top分位): {fmt(rB.get('turnover_5d'),3)}\n\n")
            tag, why = fv.recommend(rB) if rB else ("NA", "无主池数据")
            f.write(f"## 结论（基于 Universe_B）: **{tag}**\n\n{why}\n")

    # ---- master summary ----
    rows = []
    for sid, per_u in by_sig.items():
        rA, rB, rC = per_u.get("Universe_A", {}), per_u.get("Universe_B", {}), per_u.get("Universe_C", {})
        tag, why = fv.recommend(rB) if rB else ("NA", "")
        rows.append({
            "signal_id": sid, "signal_name": rB.get("signal_name", ""),
            "RankIC_5d_A": rA.get("RankIC_5d", np.nan), "RankIC_5d_B": rB.get("RankIC_5d", np.nan),
            "RankIC_5d_C": rC.get("RankIC_5d", np.nan),
            "RankICIR_5d_A": rA.get("RankICIR_5d", np.nan), "RankICIR_5d_B": rB.get("RankICIR_5d", np.nan),
            "RankICIR_5d_C": rC.get("RankICIR_5d", np.nan),
            "spread_5d_B": rB.get("spread_5d", np.nan), "costadj_spread_5d_B": rB.get("cost_adj_spread_5d", np.nan),
            "quarters_pos_B": rB.get("quarters_positive_5d", 0),
            "coverage_5d_B": rB.get("coverage_5d", np.nan),
            "decision": tag, "reason": why,
        })
    sdf = pd.DataFrame(rows).sort_values("RankICIR_5d_B", key=lambda s: s.abs(), ascending=False)
    sdf.to_csv(OUT_DIR / "cross_universe_summary.csv", index=False)

    n_keep = (sdf["decision"].isin(["保留", "重点关注"])).sum()
    n_focus = (sdf["decision"] == "重点关注").sum()
    n_drop = (sdf["decision"] == "淘汰").sum()

    with open(SUMMARY, "w") as f:
        f.write("# Alpha191 稳定性验证总结（Universe_A/B/C）\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d}  |  窗口: {START} ~ {END}\n\n")
        f.write("> label = open-to-open(T+1→T+1+h)，无未来函数；RankIC 对超额与否不敏感。\n")
        f.write("> 决策基于主池 Universe_B：|RankIC_5d|>0.015 且 |RankICIR_5d|>0.30 且 扣费后 spread 方向有利 "
                "且 ≥2 季度 RankIC 同向 且 覆盖率>80%。|RankICIR|>0.50 标记「重点关注」。\n")
        f.write("> 分年度降级为分季度（全库单一年份 2025）；分行业 N/A（无行业数据）。\n\n---\n\n")

        f.write("## 概览\n\n")
        f.write(f"| 指标 | 数值 |\n|---|---|\n")
        f.write(f"| 因子总数 | {len(sdf)} |\n")
        f.write(f"| 保留(含重点) | {n_keep} |\n")
        f.write(f"| 其中重点关注 | {n_focus} |\n")
        f.write(f"| 淘汰 | {n_drop} |\n\n")

        f.write("## 跨 Universe RankICIR_5d 对比（按 |B| 排序）\n\n")
        f.write("| Signal | Name | RankIC_5d B | ICIR_5d A | ICIR_5d B | ICIR_5d C | 扣费spread_B% | 决策 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in sdf.iterrows():
            f.write(f"| {r['signal_id']} | {r['signal_name'][:22]} | {fmt(r['RankIC_5d_B'])} | "
                    f"{fmt(r['RankICIR_5d_A'],2)} | {fmt(r['RankICIR_5d_B'],2)} | {fmt(r['RankICIR_5d_C'],2)} | "
                    f"{fmt(r['costadj_spread_5d_B'],3)} | {r['decision']} |\n")
        f.write("\n")

        # 小样本膨胀检查: A(300) vs B(731)
        f.write("## 小样本膨胀检查（A 300只 vs B 731只）\n\n")
        f.write("RankICIR 在小池 A 往往被高估。|ICIR_A| - |ICIR_B| 越大，说明扩池后衰减越明显。\n\n")
        sdf["icir_infl"] = sdf["RankICIR_5d_A"].abs() - sdf["RankICIR_5d_B"].abs()
        infl = sdf.sort_values("icir_infl", ascending=False)
        f.write("| Signal | ICIR_A | ICIR_B | 膨胀(|A|-|B|) |\n|---|---|---|---|\n")
        for _, r in infl.head(8).iterrows():
            f.write(f"| {r['signal_id']} | {fmt(r['RankICIR_5d_A'],2)} | {fmt(r['RankICIR_5d_B'],2)} | {fmt(r['icir_infl'],2)} |\n")
        med_infl = sdf["icir_infl"].median()
        f.write(f"\n> 膨胀中位数 = {med_infl:+.2f}（>0 表示小池整体高估 RankICIR）。\n\n")

        f.write("## 推荐进入 Registry 的因子\n\n")
        keep_df = sdf[sdf["decision"].isin(["保留", "重点关注"])]
        if len(keep_df):
            f.write("| Signal | Name | 决策 | RankIC_5d_B | ICIR_5d_B | 理由 |\n|---|---|---|---|---|---|\n")
            for _, r in keep_df.iterrows():
                f.write(f"| {r['signal_id']} | {r['signal_name'][:22]} | {r['decision']} | "
                        f"{fmt(r['RankIC_5d_B'])} | {fmt(r['RankICIR_5d_B'],2)} | {r['reason'][:40]} |\n")
        else:
            f.write("无因子通过主池保留标准。\n")
        f.write("\n## 说明\n\n")
        f.write("- 详见 `reports/alpha191_factor_detail/{signal_id}.md`。\n")
        f.write("- 每个 universe 的完整指标: `data/processed/signals/alpha191/{Universe}_summary.csv`。\n")
        f.write("- 「收益是否集中于少数股票」「分行业」等组合级检查在回测/融合阶段(Phase 9)补充。\n")

    print(f"\nSummary: {SUMMARY}")
    print(f"Decisions: 保留/重点={n_keep} (重点={n_focus}), 淘汰={n_drop}")


if __name__ == "__main__":
    main()
