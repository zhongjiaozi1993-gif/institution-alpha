"""Level-2 特征验证 + 对 Alpha191 增量证明（Phase 5）。

用法:
    python3 scripts/run_level2_validation.py

产出:
    data/processed/level2/level2_validation_summary.csv   # 每特征 × horizon 指标
    reports/level2_validation_report.md                   # 稳定性 + 正交性 + 增量
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.validation import level2_validator as lv

OUT_CSV = PROJECT / "data" / "processed" / "level2" / "level2_validation_summary.csv"
REPORT = PROJECT / "reports" / "level2_validation_report.md"


def fmt(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else "NA"


def main():
    fwd = lv.load_excess_fwd()
    feat_df = lv.load_l2_features()
    features = [c for c in lv.FEATURE_NAMES if c in feat_df.columns]
    print(f"Level-2 features: {len(features)} | rows {len(feat_df)} | "
          f"stocks {feat_df['symbol'].nunique()} | dates {feat_df['trade_date'].nunique()}")

    # ---- 1. 单特征稳定性 ----
    rows = [lv.validate_feature(feat_df, fwd, c) for c in features]
    sdf = pd.DataFrame(rows)
    sdf["abs_ic5"] = sdf["RankIC_5d"].abs()
    sdf = sdf.sort_values("abs_ic5", ascending=False).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sdf.drop(columns="abs_ic5").to_csv(OUT_CSV, index=False)
    print(f"Saved per-feature summary → {OUT_CSV}")

    # ---- 2. 参照 Alpha191（裁到 L2 网格）+ 最优列 ----
    alpha_wide = lv.load_alpha_on_l2_grid(feat_df)
    best_alpha_col, best_alpha_ic = lv.pick_best_alpha(alpha_wide, fwd, horizon=5)
    print(f"Best Alpha191 on L2 grid: {best_alpha_col} (RankIC_5d={best_alpha_ic:+.4f})")

    # ---- 3. Level-2 综合分（弱但正交的特征聚合）+ 正交性 + 增量 ----
    comp, comp_cols = lv.build_l2_composite(feat_df, fwd, features, horizon=5, k=6)
    ortho = lv.orthogonality(comp, alpha_wide, "l2_composite")
    inc5 = lv.incremental_test(alpha_wide, fwd, comp, best_alpha_col, horizon=5)
    inc10 = lv.incremental_test(alpha_wide, fwd, comp, best_alpha_col, horizon=10)

    _write_report(sdf, features, best_alpha_col, best_alpha_ic, comp_cols, ortho, inc5, inc10)
    print(f"Report → {REPORT}")
    print(f"L2 composite ({len(comp_cols)}): {', '.join(comp_cols)}")
    print(f"增量(5d): fused RankIC={inc5.get('rankic_fused')} vs "
          f"alpha={inc5.get('rankic_alpha')} / l2comp={inc5.get('rankic_l2comp')} | "
          f"incremental={inc5.get('incremental')}")


def _write_report(sdf, features, best_alpha_col, best_alpha_ic, comp_cols, ortho, inc5, inc10):
    n_keep = int(((sdf["RankIC_5d"].abs() > 0.015) & (sdf["RankICIR_5d"].abs() > 0.30)).sum())
    with open(REPORT, "w") as f:
        f.write("# Level-2 特征验证报告（Phase 5）\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}  |  universe: Universe_C  |  "
                f"label: open-to-open **超额**(vs 中证1000), 无未来函数\n\n")
        f.write("> 复用 Alpha191 验证管线的 IC/分位函数；fwd = label_*d_excess_index。\n")
        f.write("> 超额只影响分位绝对收益；RankIC/spread 与原始 label 完全一致。\n\n---\n\n")

        f.write("## 1. 单特征截面稳定性（按 |RankIC_5d| 排序）\n\n")
        f.write(f"覆盖标准参考 Alpha191：|RankIC_5d|>0.015 且 |RankICIR_5d|>0.30 视为有信号。"
                f"达标特征数: **{n_keep}/{len(features)}**。\n\n")
        f.write("| 特征 | 非零% | 有效日 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | 扣费spread_5d% |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in sdf.iterrows():
            f.write(f"| {r['feature']} | {r['nonzero_pct']:.0f} | {int(r['ic_n_days_5d'])} | "
                    f"{fmt(r['RankIC_1d'])} | {fmt(r['RankIC_5d'])} | {fmt(r['RankIC_10d'])} | "
                    f"{fmt(r['RankICIR_5d'],2)} | {fmt(r['cost_adj_spread_5d'],3)} |\n")

        f.write("\n## 2. 与 Alpha191 的正交性（Level-2 综合分）\n\n")
        f.write(f"Level-2 综合分 = top-{len(comp_cols)} 特征（按 |RankICIR_5d|）符号对齐等权 z 分: "
                f"**{', '.join(comp_cols)}**。\n\n与各 Alpha191 的截面 Spearman |相关| 最高者：\n\n")
        if ortho is not None and not ortho.empty:
            f.write("| Alpha191 | Spearman |\n|---|---|\n")
            for _, r in ortho.iterrows():
                f.write(f"| {r['alpha']} | {r['spearman']:+.4f} |\n")
            mx = ortho['spearman'].abs().max()
            f.write(f"\n> 最高 |相关| = {mx:.3f}，{'低' if mx < 0.3 else '中等'}，"
                    f"信息{'基本正交' if mx < 0.3 else '部分重叠'}。\n\n")
        else:
            f.write("样本不足，无法计算相关。\n\n")

        f.write("## 3. 增量证明（无 ML，IC 加权融合）\n\n")
        f.write(f"参照 Alpha191 最优列: **{best_alpha_col.replace('a_','') if best_alpha_col else 'NA'}** "
                f"(L2 网格 RankIC_5d={best_alpha_ic:+.4f})。融合 = w_a·z(alpha) + w_l·z(L2综合)，"
                f"权重 w=各自 |RankIC|。\n\n")
        f.write("| horizon | Alpha191 RankIC | L2综合 RankIC | 融合 RankIC | 融合-Alpha | 有增量 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for inc in [inc5, inc10]:
            if "error" in inc:
                f.write(f"| {inc.get('horizon','?')}d | — | — | — | — | 样本不足 |\n")
                continue
            f.write(f"| {inc['horizon']}d | {fmt(inc['rankic_alpha'])} | {fmt(inc['rankic_l2comp'])} | "
                    f"{fmt(inc['rankic_fused'])} | {fmt(inc['abs_gain_vs_alpha'])} | "
                    f"{'是' if inc['incremental'] else '否'} |\n")
        f.write("\n")
        any_inc = bool(inc5.get("incremental") or inc10.get("incremental"))
        both_inc = bool(inc5.get("incremental") and inc10.get("incremental"))
        hz = [f"{i['horizon']}d" for i in (inc5, inc10) if i.get("incremental")]
        verdict = "有增量" if both_inc else ("部分增量" if any_inc else "增量不显著")
        f.write(f"**结论: Level-2 对 Alpha191 {verdict}**")
        if any_inc:
            f.write(f"：在 {'/'.join(hz)} 上 IC 加权融合的 RankIC 高于 Alpha191 单独，"
                    "且 Level-2 综合分与 Alpha191 相关不高，说明 Level-2 携带部分正交、可增强排序的信息。"
                    "此为**样本内方向性证据**（top-k 选择/符号对齐/权重均用全样本 IC），"
                    "严格 OOS 与显著性检验留待 Phase 9（ML）。\n\n")
        else:
            f.write("：本批稀疏 Level-2 面板上，融合未稳定超越单独 Alpha191；"
                    "需更密面板/更强特征/ML 精化（Phase 9）。\n\n")

        f.write("## 4. 已知限制\n\n")
        f.write("1. 真实 Level-2 面板仅 ~27 只 × ~160 日（Universe_C 名义 175，多数无逐笔成交），"
                "截面窄、单年，RankIC 波动大，ICIR 更可信。\n")
        f.write("2. 综合分的 top-k 选择与符号对齐用全样本 IC，属**样本内**方向性证据，非 OOS。\n")
        f.write("3. 增量测试为等权/IC 加权融合（非 ML），仅作方向性证据；Phase 9 用 LightGBM 精化并做 OOS。\n")
        f.write("4. 超额基准为中证1000（idx_000852），Universe_C 多为小盘，基准合理。\n")


if __name__ == "__main__":
    main()
