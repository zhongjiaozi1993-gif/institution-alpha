"""Level-2 特征验证 + 对 Alpha191 增量证明（Phase 5 / 5.1）。

用法:
    python3 scripts/run_level2_validation.py

产出:
    data/processed/level2/level2_validation_summary.csv   # 每特征 × horizon 指标
    reports/level2_validation_report.md                   # 稳定性 + 正交性 + 样本内/OOS 增量 + 分级结论
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
TRAIN_END, TEST_START = "2025-08-31", "2025-09-01"


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
    print(f"Best Alpha191 on L2 grid (full): {best_alpha_col} (RankIC_5d={best_alpha_ic:+.4f})")

    # ---- 3. 样本内: 综合分 + 正交性(每日) + 增量 ----
    comp, comp_cols = lv.build_l2_composite(feat_df, fwd, features, horizon=5, k=6)
    ortho = lv.orthogonality(comp, alpha_wide, "l2_composite")
    inc5 = lv.incremental_test(alpha_wide, fwd, comp, best_alpha_col, horizon=5)
    inc10 = lv.incremental_test(alpha_wide, fwd, comp, best_alpha_col, horizon=10)

    # ---- 4. OOS: train 选参 / test 固定评估 ----
    oos5 = lv.oos_validation(feat_df, alpha_wide, fwd, features, horizon=5, k=6,
                             train_end=TRAIN_END, test_start=TEST_START)
    oos10 = lv.oos_validation(feat_df, alpha_wide, fwd, features, horizon=10, k=6,
                              train_end=TRAIN_END, test_start=TEST_START)

    _write_report(sdf, features, best_alpha_col, best_alpha_ic, comp_cols, ortho, inc5, inc10, oos5, oos10)
    print(f"Report → {REPORT}")
    print(f"L2 composite ({len(comp_cols)}): {', '.join(comp_cols)}")
    print(f"样本内增量(5d): fused={inc5.get('rankic_fused')} vs alpha={inc5.get('rankic_alpha')} "
          f"→ {inc5.get('incremental')}")
    if "error" not in oos5:
        print(f"OOS增量(5d): test fused={oos5['test_fused']['rankic']:.4f} vs "
              f"alpha={oos5['test_alpha']['rankic']:.4f} / l2={oos5['test_l2']['rankic']:.4f} "
              f"→ {oos5['incremental_oos']}")
    else:
        print(f"OOS(5d): {oos5['error']}")


def _fmt_inc_verdict(inc5, inc10):
    any_inc = bool(inc5.get("incremental") or inc10.get("incremental"))
    both = bool(inc5.get("incremental") and inc10.get("incremental"))
    return "有增量" if both else ("部分增量" if any_inc else "增量不显著"), any_inc


def _oos_incremental_any(oos5, oos10):
    return bool(("error" not in oos5 and oos5.get("incremental_oos")) or
                ("error" not in oos10 and oos10.get("incremental_oos")))


def _write_report(sdf, features, best_alpha_col, best_alpha_ic, comp_cols, ortho, inc5, inc10, oos5, oos10):
    n_keep = int(((sdf["RankIC_5d"].abs() > 0.015) & (sdf["RankICIR_5d"].abs() > 0.30)).sum())
    with open(REPORT, "w") as f:
        f.write("# Level-2 特征验证报告（Phase 5 / 5.1）\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}  |  universe: Universe_C  |  "
                f"label: open-to-open **超额**(vs 中证1000), 无未来函数\n\n")
        f.write("> 复用 Alpha191 验证管线的 IC/分位函数；fwd = label_*d_excess_index。\n")
        f.write("> 超额只影响分位绝对收益；RankIC/spread 与原始 label 完全一致。\n")
        f.write(f"> OOS: train ≤ {TRAIN_END}，test ≥ {TEST_START}；train 选参、test 固定评估。\n\n---\n\n")

        # ---------- 1. 单特征稳定性 ----------
        f.write("## 1. 单特征截面稳定性（按 |RankIC_5d| 排序）\n\n")
        f.write(f"|RankIC_5d|>0.015 且 |RankICIR_5d|>0.30 视为有信号。达标: **{n_keep}/{len(features)}**。\n")
        f.write("> IC有效日 = 该日截面≥5只可算 RankIC 的天数；spread有效日 = 可分五分位(≥10只)的天数。\n\n")
        f.write("| 特征 | 非零% | IC有效日 | spread有效日 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | 扣费spread_5d% |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in sdf.iterrows():
            f.write(f"| {r['feature']} | {r['nonzero_pct']:.0f} | {int(r['ic_n_days_5d'])} | "
                    f"{int(r['spread_n_days_5d'])} | {fmt(r['RankIC_1d'])} | {fmt(r['RankIC_5d'])} | "
                    f"{fmt(r['RankIC_10d'])} | {fmt(r['RankICIR_5d'],2)} | {fmt(r['cost_adj_spread_5d'],3)} |\n")

        # ---------- 2. 正交性（每日 Spearman）----------
        f.write("\n## 2. 与 Alpha191 的正交性（每日 Spearman 后汇总）\n\n")
        f.write(f"Level-2 综合分 = top-{len(comp_cols)}（|RankICIR_5d|）符号对齐等权 z 分: "
                f"**{', '.join(comp_cols)}**。\n\n")
        f.write("每日截面 Spearman(综合分, alpha) 再汇总（不再 pooled）。按 |日均| 降序取前列：\n\n")
        if ortho is not None and not ortho.empty:
            f.write("| Alpha191 | 日均 mean | median | abs_mean | 有效日 |\n|---|---|---|---|---|\n")
            for _, r in ortho.iterrows():
                f.write(f"| {r['alpha']} | {r['mean']:+.4f} | {r['median']:+.4f} | "
                        f"{r['abs_mean']:.4f} | {int(r['n_days'])} |\n")
            mx = float(ortho["abs_mean"].max())
            f.write(f"\n> 最高日均 |相关| = {mx:.3f}，{'低' if mx < 0.3 else '中等'}，"
                    f"信息{'基本正交' if mx < 0.3 else '部分重叠'}。\n\n")
        else:
            f.write("样本不足，无法计算相关。\n\n")

        # ---------- 3. 样本内增量 ----------
        f.write("## 3. 样本内增量（全样本 IC 加权融合）\n\n")
        f.write(f"参照 Alpha191 最优列: **{best_alpha_col.replace('a_','') if best_alpha_col else 'NA'}** "
                f"(全样本 RankIC_5d={best_alpha_ic:+.4f})。融合 = w_a·z(alpha)+w_l·z(L2综合)，w=|RankIC|。\n\n")
        f.write("| horizon | Alpha191 RankIC | L2综合 RankIC | 融合 RankIC | 融合-Alpha | 有增量 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for inc in [inc5, inc10]:
            if "error" in inc:
                f.write(f"| {inc.get('horizon','?')}d | — | — | — | — | 样本不足 |\n")
                continue
            f.write(f"| {inc['horizon']}d | {fmt(inc['rankic_alpha'])} | {fmt(inc['rankic_l2comp'])} | "
                    f"{fmt(inc['rankic_fused'])} | {fmt(inc['abs_gain_vs_alpha'])} | "
                    f"{'是' if inc['incremental'] else '否'} |\n")
        insample_verdict, _ = _fmt_inc_verdict(inc5, inc10)
        f.write(f"\n> 样本内结论: **{insample_verdict}**（top-k/方向/权重均用全样本 IC，有前视选择偏差）。\n\n")

        # ---------- 4. OOS 增量 ----------
        f.write("## 4. 样本外(OOS)增量（train 选参 / test 固定评估）\n\n")
        if "error" not in oos5:
            f.write(f"train ≤ {oos5['train_end']}（{oos5['n_train_dates']} 日）选出综合分特征 "
                    f"**{', '.join(oos5['chosen'])}**，最优 alpha=**{oos5['best_alpha']}**，"
                    f"权重 w_a={oos5['w_a']}/w_l={oos5['w_l']}；test ≥ {oos5['test_start']}"
                    f"（{oos5['n_test_dates']} 日）固定评估。\n\n")
            f.write("> **口径**：alpha 与 L2 综合分都按 **train 端方向**定向后再评估（committed direction，"
                    "OOS 不允许用 test 端符号），融合 = w_a·z(alpha_定向)+w_l·z(L2_定向)。表中 Alpha191 = "
                    "train-定向后的 RankIC。\n\n")
            # 参照 alpha 方向是否在 OOS 反转（train-定向 vs 未定向 raw）
            for oos in (oos5, oos10):
                da, raw = oos["test_alpha"]["rankic"], oos["test_alpha_raw_rankic"]
                flip = (not np.isnan(da)) and (not np.isnan(raw)) and (da < 0 < raw or raw < 0 < da)
                f.write(f"> {oos['horizon']}d 参照 alpha=**{oos['best_alpha']}**：train 方向 sign_a="
                        f"{oos['sign_a']:+.0f}，test 上 train-定向 RankIC={fmt(da)} / 未定向 raw={fmt(raw)}"
                        f"{'　→ **方向在 OOS 反转**（train 上有效的符号到 test 失效）' if flip else ''}。\n")
            f.write("\n")
        f.write("| horizon | 信号 | test RankIC | test RankICIR | test spread% | IC有效日 | spread有效日 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for oos in [oos5, oos10]:
            h = oos.get("horizon", "?")
            if "error" in oos:
                f.write(f"| {h}d | (样本不足) | — | — | — | — | — |\n")
                continue
            for label, ev in [("Alpha191(train定向)", oos["test_alpha"]), ("L2综合", oos["test_l2"]),
                              ("融合", oos["test_fused"])]:
                f.write(f"| {h}d | {label} | {fmt(ev['rankic'])} | {fmt(ev['rankicir'],2)} | "
                        f"{fmt(ev['spread'],3)} | {ev['ic_n_days']} | {ev['spread_n_days']} |\n")
        oos_inc_any = _oos_incremental_any(oos5, oos10)
        oos_hz = [f"{o['horizon']}d" for o in (oos5, oos10) if "error" not in o and o.get("incremental_oos")]
        l2_gen_hz = [f"{o['horizon']}d" for o in (oos5, oos10) if "error" not in o and o.get("l2_generalizes")]
        robust_hz = [f"{o['horizon']}d" for o in (oos5, oos10) if "error" not in o and o.get("robust_incremental")]
        f.write(f"\n> OOS 增量判定：**名义增量** = 融合方向泛化(RankIC>0) 且 定向融合 RankIC > 定向 Alpha191；"
                f"**稳健增量** = 名义增量 且 **同 horizon** L2 综合分自身方向泛化(避免跨 horizon 拼凑)。\n")
        f.write(f"> 名义增量 horizon: **{'/'.join(oos_hz) if oos_hz else '无'}**；"
                f"L2 综合分方向泛化 horizon: **{'/'.join(l2_gen_hz) if l2_gen_hz else '无'}**；"
                f"两者**同 horizon 同时成立（稳健增量）**: **{'/'.join(robust_hz) if robust_hz else '无'}**。\n")
        if oos_hz and not robust_hz:
            f.write("> ⚠️ 名义增量与 L2 方向泛化落在**不同 horizon**：5d 上 L2 综合分泛化但融合无增量；"
                    "10d 上融合有名义增量，但那是**赢过一个已塌缩(方向反转后近于零)的定向 alpha**、"
                    "且同 horizon L2 综合分方向未泛化。**不构成稳健 OOS 增量。**\n")
        f.write("\n")

        # ---------- 5. 分级结论 + Phase 6 建议 ----------
        _write_grading(f, inc5, inc10, oos5, oos10)

        # ---------- 6. 已知限制 ----------
        f.write("## 6. 已知限制\n\n")
        f.write("1. 面板**时间高度不均**：Universe_C 名义 175 只全部产出特征，但仅 2025-01 的 ~17 个交易日"
                "为近全池宽截面（~175 只/日），其余 ~140 日仅 ~27 只深度股；关键地，**OOS test 窗口（9–12 月）"
                "仅 26 只**。宽度集中在 train 前段，test 端仍窄——扩池主要增强样本内截面，OOS 结论仍受 test 端窄面板限制。\n")
        f.write("2. OOS 仅单一 train/test 切分（8月末），非滚动、非多折；test 仅 ~4 个月。\n")
        f.write("3. 融合为 IC 加权线性（非 ML）；Phase 9 用 LightGBM 精化并做滚动 OOS。\n")
        f.write("4. 超额基准为中证1000（idx_000852），Universe_C 多为小盘，基准合理。\n")


def _write_grading(f, inc5, inc10, oos5, oos10):
    insample_verdict, insample_any = _fmt_inc_verdict(inc5, inc10)
    oos_any = _oos_incremental_any(oos5, oos10)          # 名义增量（任一 horizon）
    nominal_hz = [f"{o['horizon']}d" for o in (oos5, oos10)
                  if "error" not in o and o.get("incremental_oos")]
    # 同 horizon 稳健增量：名义增量 且 该 horizon L2 综合分方向泛化
    robust_hz = [f"{o['horizon']}d" for o in (oos5, oos10)
                 if "error" not in o and o.get("robust_incremental")]
    oos_robust = bool(robust_hz)
    l2_gen_hz = [f"{o['horizon']}d" for o in (oos5, oos10)
                 if "error" not in o and o.get("l2_generalizes")]
    # 参照 alpha 方向是否在 OOS 反转（train-定向 与 未定向 raw 异号）
    def _flip(o):
        if "error" in o:
            return False
        da, raw = o["test_alpha"]["rankic"], o.get("test_alpha_raw_rankic", np.nan)
        return (not np.isnan(da)) and (not np.isnan(raw)) and (da < 0 < raw or raw < 0 < da)
    alpha_flip_hz = [f"{o['horizon']}d" for o in (oos5, oos10) if _flip(o)]
    recommend = oos_robust  # 仅当存在同 horizon 稳健增量才建议

    f.write("## 5. 分级结论\n\n")
    f.write("| 层级 | 结论 |\n|---|---|\n")
    f.write(f"| 样本内增量 | {insample_verdict} |\n")
    f.write(f"| OOS 名义增量（定向融合 > 定向 Alpha191） | "
            f"{('有（' + '/'.join(nominal_hz) + '）') if oos_any else '不显著'} |\n")
    f.write(f"| L2 综合分 OOS 方向泛化 | {('是（' + '/'.join(l2_gen_hz) + ' test RankIC>0）') if l2_gen_hz else '否'} |\n")
    f.write(f"| **OOS 稳健增量（同 horizon 名义增量∧L2泛化）** | **{('有（' + '/'.join(robust_hz) + '）') if oos_robust else '无'}** |\n")
    f.write(f"| 参照 Alpha191 OOS 方向稳定性 | "
            f"{('方向反转（' + '/'.join(alpha_flip_hz) + '，train 符号到 test 失效）') if alpha_flip_hz else '未见反转'} |\n")
    f.write(f"| **是否建议进入 Phase 6 规则融合** | **{'建议（小权重增强项）' if recommend else '暂不建议（先纯 Alpha191 融合）'}** |\n\n")

    if alpha_flip_hz:
        f.write(f"> 注意：样本内最优参照 Alpha191 在 OOS（{'/'.join(alpha_flip_hz)}）方向反转——"
                "在 train 上有效的符号到 test 已失效。这正是 OOS 要暴露的“样本内选优”陷阱："
                "样本内增量高度依赖那个不稳定的 alpha，不可直接外推。\n\n")

    if recommend:
        f.write(f"> 建议：Level-2 综合分在 OOS（{'/'.join(robust_hz)}）上同 horizon 既方向泛化又对定向 Alpha191 有增量，"
                "可进入 Phase 6，将 Level-2 综合分作为 Alpha191 排序的**小权重增强项**"
                "（先小仓验证，避免因面板窄而过配）。\n\n")
    elif oos_any and not oos_robust:
        f.write("> 建议：OOS 仅有**名义增量而无同 horizon 稳健增量**（名义增量落在 L2 未泛化的 horizon、"
                "且赢过的是已塌缩的定向 alpha），**暂不进入** Phase 6 的 Level-2 融合。"
                "Phase 6 先做**纯 Alpha191 规则融合**；Level-2 待扩面板（更多股票/更密日历）或 Phase 9 ML+滚动 OOS 再评估。\n\n")
    elif insample_any and not oos_any:
        f.write("> 建议：样本内有增量但 OOS 未确认，**暂不进入** Phase 6 的 Level-2 融合；"
                "优先扩充 Level-2 面板（更多股票/更密日历），或等 Phase 9 ML + 滚动 OOS 再评估。"
                "Phase 6 可先只做纯 Alpha191 规则融合。\n\n")
    else:
        f.write("> 建议：样本内与 OOS 均未确认稳健增量，**暂不进入** Phase 6 的 Level-2 融合；"
                "先扩面板/改特征，Phase 6 以纯 Alpha191 规则融合为主。\n\n")


if __name__ == "__main__":
    main()
