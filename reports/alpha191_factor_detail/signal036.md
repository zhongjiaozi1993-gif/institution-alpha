# Signal036: Alpha191_VolPrice_GTJA150 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | -0.0204 | -0.0387 | -0.0505 | -0.30 | -0.446 | -0.966 | 97.5 |
| Universe_B | 731 | -0.0128 | -0.0123 | -0.0145 | -0.08 | 0.057 | -0.463 | 97.5 |
| Universe_C | 160 | -0.0103 | -0.0303 | -0.0398 | -0.25 | -0.510 | -1.030 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.0611, Q2=-0.0552, Q3=-0.0014, Q4=-0.0422
- Universe_B: Q1=-0.0193, Q2=-0.0340, Q3=+0.0109, Q4=-0.0093
- Universe_C: Q1=-0.0362, Q2=-0.0508, Q3=+0.0140, Q4=-0.0554

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=-0.0111, mid=-0.0156, high=-0.0149
- 流动性组: low=-0.0044, mid=-0.0147, high=-0.0399
- 换手率(5d top分位): 0.025

## 结论（基于 Universe_B）: **淘汰**

|RankIC_5d|=0.012≤0.015; |RankICIR_5d|=0.08≤0.30
