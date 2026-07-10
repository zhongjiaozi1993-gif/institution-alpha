# Signal039: Alpha191_Volatility_GTJA076 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | -0.0017 | 0.0016 | -0.0023 | 0.02 | 0.067 | -0.453 | 97.5 |
| Universe_B | 731 | -0.0018 | 0.0007 | -0.0006 | 0.01 | 0.174 | -0.346 | 97.5 |
| Universe_C | 160 | -0.0049 | -0.0090 | -0.0124 | -0.08 | -0.032 | -0.552 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.0187, Q2=+0.0608, Q3=-0.0442, Q4=+0.0058
- Universe_B: Q1=-0.0220, Q2=+0.0390, Q3=-0.0158, Q4=-0.0062
- Universe_C: Q1=-0.0668, Q2=+0.0447, Q3=-0.0143, Q4=-0.0225

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=-0.0049, mid=+0.0069, high=-0.0017
- 流动性组: low=-0.0042, mid=-0.0049, high=+0.0088
- 换手率(5d top分位): 0.159

## 结论（基于 Universe_B）: **淘汰**

|RankIC_5d|=0.001≤0.015; |RankICIR_5d|=0.01≤0.30; 扣费后 spread 方向不利(-0.35%); 仅1季度 RankIC>0
