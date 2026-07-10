# Signal042: Alpha191_Volatility_GTJA161 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | -0.0375 | -0.0614 | -0.0773 | -0.35 | -0.330 | -0.850 | 97.5 |
| Universe_B | 731 | -0.0258 | -0.0334 | -0.0365 | -0.18 | 0.101 | -0.419 | 97.5 |
| Universe_C | 160 | -0.0294 | -0.0565 | -0.0709 | -0.35 | -0.397 | -0.917 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.1004, Q2=-0.0763, Q3=-0.0248, Q4=-0.0563
- Universe_B: Q1=-0.0776, Q2=-0.0476, Q3=-0.0001, Q4=-0.0206
- Universe_C: Q1=-0.0700, Q2=-0.0806, Q3=-0.0103, Q4=-0.0747

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=-0.0372, mid=-0.0368, high=-0.0330
- 流动性组: low=-0.0291, mid=-0.0354, high=-0.0596
- 换手率(5d top分位): 0.030

## 结论（基于 Universe_B）: **淘汰**

|RankICIR_5d|=0.18≤0.30
