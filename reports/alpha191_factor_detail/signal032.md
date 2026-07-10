# Signal032: Alpha191_VolPrice_GTJA032 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | 0.0079 | -0.0016 | 0.0031 | -0.01 | -0.019 | -0.539 | 97.5 |
| Universe_B | 731 | 0.0070 | 0.0004 | 0.0069 | 0.00 | -0.103 | -0.623 | 97.5 |
| Universe_C | 160 | 0.0182 | 0.0025 | 0.0073 | 0.02 | 0.019 | -0.501 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.0069, Q2=+0.0258, Q3=-0.0105, Q4=-0.0163
- Universe_B: Q1=+0.0081, Q2=-0.0001, Q3=-0.0067, Q4=+0.0024
- Universe_C: Q1=-0.0135, Q2=+0.0255, Q3=+0.0055, Q4=-0.0120

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=+0.0006, mid=-0.0065, high=+0.0043
- 流动性组: low=-0.0011, mid=+0.0078, high=-0.0024
- 换手率(5d top分位): 0.651

## 结论（基于 Universe_B）: **淘汰**

|RankIC_5d|=0.000≤0.015; |RankICIR_5d|=0.00≤0.30; 扣费后 spread 方向不利(-0.62%)
