# Alpha191 稳定性验证总结（Universe_A/B/C）

生成时间: 2026-07-10  |  窗口: 2025-01-01 ~ 2025-12-31

> label = open-to-open(T+1→T+1+h)，无未来函数；RankIC 对超额与否不敏感。
> 决策基于主池 Universe_B：|RankIC_5d|>0.015 且 |RankICIR_5d|>0.30 且 扣费后 spread 方向有利 且 ≥2 季度 RankIC 同向 且 覆盖率>80%。|RankICIR|>0.50 标记「重点关注」。
> 分年度降级为分季度（全库单一年份 2025）；分行业 N/A（无行业数据）。

---

## 概览

| 指标 | 数值 |
|---|---|
| 因子总数 | 30 |
| 保留(含重点) | 11 |
| 其中重点关注 | 2 |
| 淘汰 | 19 |

## 跨 Universe RankICIR_5d 对比（按 |B| 排序）

| Signal | Name | RankIC_5d B | ICIR_5d A | ICIR_5d B | ICIR_5d C | 扣费spread_B% | 决策 |
|---|---|---|---|---|---|---|---|
| Signal040 | Alpha191_Volatility_GT | -0.0944 | -0.64 | -0.55 | -0.64 | -1.180 | 重点关注 |
| Signal019 | Alpha191_Volatility_GT | -0.0880 | -0.60 | -0.53 | -0.66 | -0.997 | 重点关注 |
| Signal033 | Alpha191_VolPrice_GTJA | -0.0518 | -0.59 | -0.49 | -0.52 | -0.774 | 保留 |
| Signal027 | Alpha191_Reversal_GTJA | 0.0654 | 0.47 | 0.47 | 0.44 | -0.205 | 淘汰 |
| Signal023 | Alpha191_Momentum_GTJA | -0.0700 | -0.48 | -0.45 | -0.47 | -0.683 | 保留 |
| Signal024 | Alpha191_Momentum_GTJA | -0.0637 | -0.46 | -0.42 | -0.42 | -0.694 | 保留 |
| Signal025 | Alpha191_Momentum_GTJA | -0.0471 | -0.41 | -0.37 | -0.40 | -0.585 | 保留 |
| Signal022 | Alpha191_Momentum_GTJA | -0.0368 | -0.45 | -0.37 | -0.40 | -1.139 | 保留 |
| Signal030 | Alpha191_Reversal_GTJA | -0.0418 | -0.40 | -0.36 | -0.31 | -0.599 | 保留 |
| Signal038 | Alpha191_Volatility_GT | 0.0389 | 0.37 | 0.36 | 0.48 | -0.481 | 淘汰 |
| Signal035 | Alpha191_VolPrice_GTJA | -0.0416 | -0.40 | -0.34 | -0.43 | -0.519 | 保留 |
| Signal029 | Alpha191_Reversal_GTJA | -0.0432 | -0.29 | -0.33 | -0.33 | -0.741 | 保留 |
| Signal028 | Alpha191_Reversal_GTJA | 0.0432 | 0.29 | 0.33 | 0.33 | -0.300 | 淘汰 |
| Signal021 | Alpha191_Momentum_GTJA | -0.0410 | -0.28 | -0.32 | -0.27 | -0.679 | 保留 |
| Signal034 | Alpha191_VolPrice_GTJA | -0.0256 | -0.33 | -0.27 | -0.34 | -0.467 | 淘汰 |
| Signal041 | Alpha191_Volatility_GT | -0.0449 | -0.38 | -0.26 | -0.36 | -0.081 | 淘汰 |
| Signal026 | Alpha191_Momentum_GTJA | -0.0452 | -0.42 | -0.24 | -0.45 | -0.477 | 淘汰 |
| Signal044 | Alpha191_Trend_GTJA096 | -0.0302 | -0.25 | -0.24 | -0.23 | -0.508 | 淘汰 |
| Signal020 | Alpha191_Momentum_GTJA | 0.0226 | 0.26 | 0.23 | 0.11 | -0.080 | 淘汰 |
| Signal031 | Alpha191_VolPrice_GTJA | -0.0236 | -0.22 | -0.23 | -0.15 | -0.741 | 淘汰 |
| Signal046 | Alpha191_Trend_GTJA172 | -0.0179 | -0.13 | -0.19 | -0.22 | -0.466 | 淘汰 |
| Signal042 | Alpha191_Volatility_GT | -0.0334 | -0.35 | -0.18 | -0.35 | -0.419 | 淘汰 |
| Signal043 | Alpha191_Trend_GTJA089 | -0.0231 | -0.10 | -0.16 | -0.15 | -0.572 | 淘汰 |
| Signal037 | Alpha191_VolPrice_GTJA | -0.0199 | -0.13 | -0.15 | -0.15 | -0.644 | 淘汰 |
| Signal045 | Alpha191_Trend_GTJA153 | -0.0166 | -0.29 | -0.11 | -0.18 | -0.582 | 淘汰 |
| Signal018 | Alpha191_VolumePrice_G | 0.0080 | 0.06 | 0.11 | 0.13 | NA | 淘汰 |
| Signal017 | Alpha191_Reversal_GTJA | 0.0104 | 0.07 | 0.08 | 0.05 | -0.361 | 淘汰 |
| Signal036 | Alpha191_VolPrice_GTJA | -0.0123 | -0.30 | -0.08 | -0.25 | -0.463 | 淘汰 |
| Signal039 | Alpha191_Volatility_GT | 0.0007 | 0.02 | 0.01 | -0.08 | -0.346 | 淘汰 |
| Signal032 | Alpha191_VolPrice_GTJA | 0.0004 | -0.01 | 0.00 | 0.02 | -0.623 | 淘汰 |

## 小样本膨胀检查（A 300只 vs B 731只）

RankICIR 在小池 A 往往被高估。|ICIR_A| - |ICIR_B| 越大，说明扩池后衰减越明显。

| Signal | ICIR_A | ICIR_B | 膨胀(|A|-|B|) |
|---|---|---|---|
| Signal036 | -0.30 | -0.08 | 0.22 |
| Signal026 | -0.42 | -0.24 | 0.18 |
| Signal045 | -0.29 | -0.11 | 0.18 |
| Signal042 | -0.35 | -0.18 | 0.17 |
| Signal041 | -0.38 | -0.26 | 0.12 |
| Signal033 | -0.59 | -0.49 | 0.10 |
| Signal040 | -0.64 | -0.55 | 0.09 |
| Signal022 | -0.45 | -0.37 | 0.08 |

> 膨胀中位数 = +0.03（>0 表示小池整体高估 RankICIR）。

## 推荐进入 Registry 的因子

| Signal | Name | 决策 | RankIC_5d_B | ICIR_5d_B | 理由 |
|---|---|---|---|---|---|
| Signal040 | Alpha191_Volatility_GT | 重点关注 | -0.0944 | -0.55 | 反向有效; 全部达标 |
| Signal019 | Alpha191_Volatility_GT | 重点关注 | -0.0880 | -0.53 | 反向有效; 全部达标 |
| Signal033 | Alpha191_VolPrice_GTJA | 保留 | -0.0518 | -0.49 | 反向有效; 全部达标 |
| Signal023 | Alpha191_Momentum_GTJA | 保留 | -0.0700 | -0.45 | 反向有效; 全部达标 |
| Signal024 | Alpha191_Momentum_GTJA | 保留 | -0.0637 | -0.42 | 反向有效; 全部达标 |
| Signal025 | Alpha191_Momentum_GTJA | 保留 | -0.0471 | -0.37 | 反向有效; 全部达标 |
| Signal022 | Alpha191_Momentum_GTJA | 保留 | -0.0368 | -0.37 | 反向有效; 全部达标 |
| Signal030 | Alpha191_Reversal_GTJA | 保留 | -0.0418 | -0.36 | 反向有效; 全部达标 |
| Signal035 | Alpha191_VolPrice_GTJA | 保留 | -0.0416 | -0.34 | 反向有效; 全部达标 |
| Signal029 | Alpha191_Reversal_GTJA | 保留 | -0.0432 | -0.33 | 反向有效; 全部达标 |
| Signal021 | Alpha191_Momentum_GTJA | 保留 | -0.0410 | -0.32 | 反向有效; 全部达标 |

## 说明

- 详见 `reports/alpha191_factor_detail/{signal_id}.md`。
- 每个 universe 的完整指标: `data/processed/signals/alpha191/{Universe}_summary.csv`。
- 「收益是否集中于少数股票」「分行业」等组合级检查在回测/融合阶段(Phase 9)补充。
