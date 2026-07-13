# Phase 7A 长期信号衰减验证报告

生成时间: 2026-07-13 21:01  |  主方案=equal_weight, 对照=best_single  |  horizon=[5, 10, 20, 40, 60], embargo=6

> **目的**：验证 Alpha191 的稳定排序信息是否能延伸到 20d/40d/60d。
> 长周期若无效，Phase 7A 后半段低换手引擎暂停，不继续堆规则。
> 每折在其**自己的 train** 上独立确定因子/方向/权重；逐 horizon 独立 purge；test 冻结。
> 合格门槛**预固定**（不看完结果再定），见文末。

---

## 衰减曲线：跨 horizon RankIC / RankICIR

| horizon | expanding pooled RankIC | expanding pooled RankICIR | rolling pooled RankIC | rolling pooled RankICIR | 合格? |
|---|---|---|---|---|---|
| 5d | +0.061 | +0.478 | +0.064 | +0.508 | exp=✓ roll=✓ |
| 10d | +0.072 | +0.603 | +0.067 | +0.589 | exp=✓ roll=✓ |
| 20d | +0.058 | +0.487 | +0.037 | +0.416 | exp=✗ roll=✗ |
| 40d | +0.065 | +0.587 | +0.064 | +0.565 | exp=✗ roll=✗ |
| 60d | +0.052 | +0.484 | +0.027 | +0.263 | exp=✗ roll=✗ |


## 持有 5d

### expanding window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 1 | +0.042 | +0.294 | +51% | +22.02% | +77.54% | +55.52% | +0.042 | +0.294 |
| 2025-07 | 2025-01→2025-06 | 2025-07→2025-08 | 3 | +0.075 | +0.643 | +84% | +15.30% | +196.66% | +181.37% | +0.062 | +0.483 |
| 2025-09 | 2025-01→2025-08 | 2025-09→2025-10 | 4 | +0.057 | +0.433 | +64% | +15.44% | +70.83% | +55.39% | +0.088 | +0.453 |
| 2025-11 | 2025-01→2025-10 | 2025-11→2025-12 | 6 | +0.067 | +0.550 | +72% | -25.90% | +87.24% | +113.14% | +0.031 | +0.209 |

**expanding 聚合（4/4 折）**：
- RankIC 正折比: +100%  |  pooled RankIC: +0.061  |  pooled RankICIR: +0.478
- RankICIR 跨折: [+0.29, +0.64] 均值 +0.48  |  最差折 RankIC: +0.042
- 分位 spread 均值: +6.71%  |  spread 正比率: +75%
- best_single RankIC 均值: +0.056  |  best_single RankICIR 均值: +2.578
- 入选因子数: 1–6  |  kept 相邻 Jaccard: +0.39  |  方向翻转: 无 (+0%)
- 因子入选频率: signal040×3, signal033×3, signal023×3, signal028×1, signal029×1, signal027×1, signal022×1, signal026×1

**预固定门槛判定**: ✓ 通过
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✓
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✓

### rolling window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 1 | +0.042 | +0.294 | +51% | +22.02% | +77.54% | +55.52% | +0.042 | +0.294 |
| 2025-07 | 2025-03→2025-06 | 2025-07→2025-08 | 6 | +0.059 | +0.438 | +77% | -26.36% | +170.00% | +196.36% | +0.069 | +0.533 |
| 2025-09 | 2025-05→2025-08 | 2025-09→2025-10 | 9 | +0.074 | +0.726 | +79% | +44.12% | +81.38% | +37.26% | +0.011 | +0.120 |
| 2025-11 | 2025-07→2025-10 | 2025-11→2025-12 | 11 | +0.079 | +0.656 | +65% | +9.12% | +115.93% | +106.81% | +0.066 | +0.545 |

**rolling 聚合（4/4 折）**：
- RankIC 正折比: +100%  |  pooled RankIC: +0.064  |  pooled RankICIR: +0.508
- RankICIR 跨折: [+0.29, +0.73] 均值 +0.53  |  最差折 RankIC: +0.042
- 分位 spread 均值: +12.22%  |  spread 正比率: +75%
- best_single RankIC 均值: +0.047  |  best_single RankICIR 均值: +2.035
- 入选因子数: 1–11  |  kept 相邻 Jaccard: +0.34  |  方向翻转: 无 (+0%)
- 因子入选频率: signal019×3, signal033×3, signal024×3, signal020×2, signal046×2, signal027×2, signal034×2, signal031×2, signal043×2, signal028×1, signal026×1, signal041×1, signal038×1, signal035×1, signal044×1

**预固定门槛判定**: ✓ 通过
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✓
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✓

## 持有 10d

### expanding window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 3 | +0.041 | +0.468 | +69% | +61.00% | +222.02% | +161.03% | +0.156 | +1.717 |
| 2025-07 | 2025-01→2025-06 | 2025-07→2025-08 | 2 | +0.082 | +0.783 | +73% | +51.33% | +395.16% | +343.83% | +0.070 | +0.555 |
| 2025-09 | 2025-01→2025-08 | 2025-09→2025-10 | 4 | +0.078 | +0.527 | +69% | +74.62% | +154.10% | +79.47% | +0.133 | +0.735 |
| 2025-11 | 2025-01→2025-10 | 2025-11→2025-12 | 6 | +0.084 | +0.651 | +67% | +8.27% | +259.97% | +251.70% | -0.002 | -0.013 |

**expanding 聚合（4/4 折）**：
- RankIC 正折比: +100%  |  pooled RankIC: +0.072  |  pooled RankICIR: +0.603
- RankICIR 跨折: [+0.47, +0.78] 均值 +0.61  |  最差折 RankIC: +0.041
- 分位 spread 均值: +48.80%  |  spread 正比率: +100%
- best_single RankIC 均值: +0.089  |  best_single RankICIR 均值: +1.457
- 入选因子数: 2–6  |  kept 相邻 Jaccard: +0.39  |  方向翻转: 无 (+0%)
- 因子入选频率: signal040×3, signal033×3, signal023×2, signal027×2, signal019×1, signal034×1, signal044×1, signal035×1, signal022×1

**预固定门槛判定**: ✓ 通过
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✓
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✓

### rolling window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 3 | +0.041 | +0.468 | +69% | +61.00% | +222.02% | +161.03% | +0.156 | +1.717 |
| 2025-07 | 2025-03→2025-06 | 2025-07→2025-08 | 7 | +0.054 | +0.449 | +68% | -24.90% | +349.94% | +374.84% | +0.068 | +0.512 |
| 2025-09 | 2025-05→2025-08 | 2025-09→2025-10 | 7 | +0.094 | +0.769 | +77% | +108.17% | +187.19% | +79.02% | +0.133 | +0.725 |
| 2025-11 | 2025-07→2025-10 | 2025-11→2025-12 | 10 | +0.081 | +0.669 | +67% | +17.06% | +263.88% | +246.82% | +0.067 | +0.657 |

**rolling 聚合（4/4 折）**：
- RankIC 正折比: +100%  |  pooled RankIC: +0.067  |  pooled RankICIR: +0.589
- RankICIR 跨折: [+0.45, +0.77] 均值 +0.59  |  最差折 RankIC: +0.041
- 分位 spread 均值: +40.33%  |  spread 正比率: +75%
- best_single RankIC 均值: +0.106  |  best_single RankICIR 均值: +2.717
- 入选因子数: 3–10  |  kept 相邻 Jaccard: +0.31  |  方向翻转: ['signal034'] (+7%)
- 因子入选频率: signal019×3, signal034×3, signal033×3, signal020×3, signal035×2, signal046×2, signal043×2, signal027×2, signal044×1, signal026×1, signal041×1, signal023×1, signal040×1, signal038×1, signal031×1

**预固定门槛判定**: ✓ 通过
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✓
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✓

## 持有 20d

### expanding window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 3 | +0.096 | +1.160 | +82% | +188.82% | +554.57% | +365.75% | +0.149 | +1.910 |
| 2025-07 | 2025-01→2025-06 | 2025-07→2025-08 | 4 | -0.017 | -0.257 | +36% | -241.52% | +615.02% | +856.54% | +0.046 | +0.599 |
| 2025-09 | 2025-01→2025-08 | 2025-09→2025-10 | 7 | +0.126 | +1.046 | +85% | +194.08% | +187.42% | -6.66% | +0.051 | +0.712 |
| 2025-11 | 2025-01→2025-10 | 2025-11→2025-12 | 6 | +0.037 | +0.267 | +47% | -178.88% | +571.06% | +749.94% | -0.037 | -0.306 |

**expanding 聚合（4/4 折）**：
- RankIC 正折比: +75%  |  pooled RankIC: +0.058  |  pooled RankICIR: +0.487
- RankICIR 跨折: [-0.26, +1.16] 均值 +0.55  |  最差折 RankIC: -0.017
- 分位 spread 均值: -9.37%  |  spread 正比率: +50%
- best_single RankIC 均值: +0.052  |  best_single RankICIR 均值: +0.787
- 入选因子数: 3–7  |  kept 相邻 Jaccard: +0.47  |  方向翻转: 无 (+0%)
- 因子入选频率: signal035×3, signal022×3, signal019×2, signal026×2, signal033×2, signal040×2, signal023×2, signal027×2, signal034×1, signal036×1

**预固定门槛判定**: ✗ 未通过（未过: g4_worst_fold, g5_quantile_consistent, g6_not_single_fold）
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✗
  - g5_quantile_consistent: ✗
  - g6_not_single_fold: ✗

### rolling window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 3 | +0.096 | +1.160 | +82% | +188.82% | +554.57% | +365.75% | +0.149 | +1.910 |
| 2025-07 | 2025-03→2025-06 | 2025-07→2025-08 | 11 | -0.022 | -0.324 | +43% | -266.20% | +592.90% | +859.10% | +0.046 | +0.599 |
| 2025-09 | 2025-05→2025-08 | 2025-09→2025-10 | 11 | +0.056 | +0.565 | +69% | +58.26% | +123.39% | +65.13% | +0.051 | +0.712 |
| 2025-11 | 2025-07→2025-10 | 2025-11→2025-12 | 9 | +0.025 | +0.425 | +70% | +67.88% | +632.56% | +564.68% | +0.036 | +0.334 |

**rolling 聚合（4/4 折）**：
- RankIC 正折比: +75%  |  pooled RankIC: +0.037  |  pooled RankICIR: +0.416
- RankICIR 跨折: [-0.32, +1.16] 均值 +0.46  |  最差折 RankIC: -0.022
- 分位 spread 均值: +12.19%  |  spread 正比率: +75%
- best_single RankIC 均值: +0.070  |  best_single RankICIR 均值: +1.541
- 入选因子数: 3–11  |  kept 相邻 Jaccard: +0.23  |  方向翻转: ['signal034', 'signal024', 'signal022', 'signal046'] (+20%)
- 因子入选频率: signal019×4, signal033×3, signal046×3, signal034×2, signal024×2, signal035×2, signal022×2, signal043×2, signal021×2, signal039×2, signal036×1, signal026×1, signal041×1, signal027×1, signal038×1, signal030×1, signal031×1, signal045×1, signal020×1, signal044×1

**预固定门槛判定**: ✗ 未通过（未过: g4_worst_fold, g6_not_single_fold）
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✗
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✗

## 持有 40d

### expanding window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 2 | +0.064 | +0.777 | +82% | +73.84% | +1401.54% | +1327.71% | +0.102 | +1.026 |
| 2025-07 | 2025-01→2025-06 | 2025-07→2025-08 | 4 | +0.027 | +0.267 | +55% | -137.00% | +1088.04% | +1225.04% | +0.101 | +0.897 |
| 2025-09 | 2025-01→2025-08 | 2025-09→2025-10 | 6 | +0.154 | +1.241 | +90% | +398.87% | +307.75% | -91.13% | +0.228 | +1.972 |
| 2025-11 | 2025-01→2025-10 | 2025-11→2025-12 | 7 | +0.022 | +0.281 | +58% | -424.36% | +1230.47% | +1654.83% | -0.013 | -0.245 |

**expanding 聚合（4/4 折）**：
- RankIC 正折比: +100%  |  pooled RankIC: +0.065  |  pooled RankICIR: +0.587
- RankICIR 跨折: [+0.27, +1.24] 均值 +0.64  |  最差折 RankIC: +0.022
- 分位 spread 均值: -22.16%  |  spread 正比率: +50%
- best_single RankIC 均值: +0.104  |  best_single RankICIR 均值: +1.223
- 入选因子数: 2–7  |  kept 相邻 Jaccard: +0.31  |  方向翻转: 无 (+0%)
- 因子入选频率: signal019×4, signal038×3, signal026×2, signal035×2, signal033×2, signal037×1, signal024×1, signal030×1, signal023×1, signal039×1, signal042×1

**预固定门槛判定**: ✗ 未通过（未过: g5_quantile_consistent）
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✓
  - g5_quantile_consistent: ✗
  - g6_not_single_fold: ✓

### rolling window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 2 | +0.064 | +0.777 | +82% | +73.84% | +1401.54% | +1327.71% | +0.102 | +1.026 |
| 2025-07 | 2025-03→2025-06 | 2025-07→2025-08 | 5 | +0.045 | +0.368 | +55% | +36.17% | +1094.79% | +1058.61% | +0.101 | +0.897 |
| 2025-09 | 2025-05→2025-08 | 2025-09→2025-10 | 3 | +0.168 | +1.665 | +95% | +522.53% | +320.97% | -201.56% | +0.252 | +2.559 |
| 2025-11 | 2025-07→2025-10 | 2025-11→2025-12 | 7 | -0.009 | -0.127 | +51% | -136.68% | +1290.31% | +1426.99% | +0.019 | +0.646 |

**rolling 聚合（4/4 折）**：
- RankIC 正折比: +75%  |  pooled RankIC: +0.064  |  pooled RankICIR: +0.565
- RankICIR 跨折: [-0.13, +1.67] 均值 +0.67  |  最差折 RankIC: -0.009
- 分位 spread 均值: +123.97%  |  spread 正比率: +75%
- best_single RankIC 均值: +0.118  |  best_single RankICIR 均值: +1.407
- 入选因子数: 2–7  |  kept 相邻 Jaccard: +0.19  |  方向翻转: ['signal046'] (+8%)
- 因子入选频率: signal033×3, signal019×2, signal046×2, signal040×2, signal037×1, signal036×1, signal023×1, signal024×1, signal039×1, signal022×1, signal020×1, signal044×1

**预固定门槛判定**: ✗ 未通过（未过: g6_not_single_fold）
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✓
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✗

## 持有 60d

### expanding window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 2 | -0.050 | -0.734 | +18% | -148.03% | +2092.86% | +2240.89% | -0.057 | -1.058 |
| 2025-07 | 2025-01→2025-06 | 2025-07→2025-08 | 4 | +0.095 | +0.789 | +70% | +248.32% | +1317.26% | +1068.94% | +0.158 | +1.682 |
| 2025-09 | 2025-01→2025-08 | 2025-09→2025-10 | 6 | +0.111 | +1.393 | +90% | +288.64% | +918.63% | +629.99% | +0.152 | +1.239 |
| 2025-11 | 2025-01→2025-10 | 2025-11→2025-12 | 4 | +0.045 | +0.627 | +81% | -42.74% | +1332.36% | +1375.10% | +0.040 | +0.584 |

**expanding 聚合（4/4 折）**：
- RankIC 正折比: +75%  |  pooled RankIC: +0.052  |  pooled RankICIR: +0.484
- RankICIR 跨折: [-0.73, +1.39] 均值 +0.52  |  最差折 RankIC: -0.050
- 分位 spread 均值: +86.55%  |  spread 正比率: +50%
- best_single RankIC 均值: +0.073  |  best_single RankICIR 均值: +0.824
- 入选因子数: 2–6  |  kept 相邻 Jaccard: +0.29  |  方向翻转: 无 (+0%)
- 因子入选频率: signal036×3, signal019×3, signal021×3, signal037×1, signal041×1, signal035×1, signal022×1, signal034×1, signal033×1, signal024×1

**预固定门槛判定**: ✗ 未通过（未过: g4_worst_fold, g5_quantile_consistent, g6_not_single_fold）
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✗
  - g5_quantile_consistent: ✗
  - g6_not_single_fold: ✗

### rolling window

| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | IC正日比 | 分位spread | top均值 | bot均值 | 对照RankIC | 对照RankICIR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-05 | 2025-01→2025-04 | 2025-05→2025-06 | 2 | -0.050 | -0.734 | +18% | -148.03% | +2092.86% | +2240.89% | -0.057 | -1.058 |
| 2025-07 | 2025-03→2025-06 | 2025-07→2025-08 | 11 | +0.067 | +0.501 | +68% | +111.48% | +1208.93% | +1097.45% | +0.159 | +1.701 |
| 2025-09 | 2025-05→2025-08 | 2025-09→2025-10 | 10 | +0.059 | +0.731 | +77% | +168.81% | +881.35% | +712.54% | +0.152 | +1.239 |
| 2025-11 | 2025-07→2025-10 | 2025-11→2025-12 | 10 | +0.025 | +0.416 | +58% | +270.02% | +1483.57% | +1213.54% | +0.040 | +0.584 |

**rolling 聚合（4/4 折）**：
- RankIC 正折比: +75%  |  pooled RankIC: +0.027  |  pooled RankICIR: +0.263
- RankICIR 跨折: [-0.73, +0.73] 均值 +0.23  |  最差折 RankIC: -0.050
- 分位 spread 均值: +100.57%  |  spread 正比率: +75%
- best_single RankIC 均值: +0.073  |  best_single RankICIR 均值: +0.824
- 入选因子数: 2–11  |  kept 相邻 Jaccard: +0.14  |  方向翻转: ['signal041', 'signal031', 'signal024', 'signal034', 'signal030'] (+25%)
- 因子入选频率: signal034×3, signal030×3, signal036×2, signal037×2, signal041×2, signal039×2, signal033×2, signal031×2, signal024×2, signal043×2, signal019×2, signal040×1, signal022×1, signal021×1, signal023×1, signal046×1, signal035×1, signal042×1, signal038×1, signal045×1

**预固定门槛判定**: ✗ 未通过（未过: g4_worst_fold, g6_not_single_fold）
  - g1_pos_folds: ✓
  - g2_pooled_ric_pos: ✓
  - g3_pooled_ricir: ✓
  - g4_worst_fold: ✗
  - g5_quantile_consistent: ✓
  - g6_not_single_fold: ✗

---

## 跨 horizon 因子稳定性

| horizon | expanding 入选数范围 | expanding Jaccard | rolling 入选数范围 | rolling Jaccard | 翻转因子数 |
|---|---|---|---|---|---|
| 5d | 1–6 | +0.39 | 1–11 | +0.34 | 0 |
| 10d | 2–6 | +0.39 | 3–10 | +0.31 | 1 |
| 20d | 3–7 | +0.47 | 3–11 | +0.23 | 4 |
| 40d | 2–7 | +0.31 | 2–7 | +0.19 | 1 |
| 60d | 2–6 | +0.29 | 2–11 | +0.14 | 5 |

---

## 合格 horizon 汇总

| horizon | expanding | rolling | 综合判定 |
|---|---|---|---|
| 5d | ✓ | ✓ | ✓✓ 双窗通过 |
| 10d | ✓ | ✓ | ✓✓ 双窗通过 |
| 20d | ✗ | ✗ | ✗ 未通过 |
| 40d | ✗ | ✗ | ✗ 未通过 |
| 60d | ✗ | ✗ | ✗ 未通过 |

### 预固定合格门槛定义

1. 至少 3/4 fold RankIC > 0
2. 合并 OOS RankIC > 0.0
3. 合并 RankICIR > 0.25
4. 最差 fold RankIC ≥ -0.01
5. 高分位相对低分位方向与 RankIC 一致（≥3/4 折 spread > 0）
6. 不依赖单一 fold（ICIR 跨折极差 < 2×|均值|）
7. 因子方向翻转率不显著高于 5d/10d 基准（在衰减曲线表格中对比）

### 下一步决策

- **有合格长周期（20d+）**：继续 Phase 7A Commit 2 低换手状态机。
- **仅 20d 勉强有效**：缩小 Phase 7A，只做 20d 低换手。
- **20/40/60d 全部失效**：停止死磕 Alpha191，转入 Phase 8 多源因子工厂。
