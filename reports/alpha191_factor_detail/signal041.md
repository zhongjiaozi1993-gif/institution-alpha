# Signal041: Alpha191_Volatility_GTJA158 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | -0.0445 | -0.0623 | -0.0708 | -0.38 | 0.175 | -0.345 | 97.5 |
| Universe_B | 731 | -0.0368 | -0.0449 | -0.0485 | -0.26 | 0.439 | -0.081 | 97.5 |
| Universe_C | 160 | -0.0474 | -0.0663 | -0.0713 | -0.36 | 0.167 | -0.353 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.0427, Q2=-0.0709, Q3=-0.0688, Q4=-0.0658
- Universe_B: Q1=-0.0349, Q2=-0.0569, Q3=-0.0371, Q4=-0.0516
- Universe_C: Q1=-0.0347, Q2=-0.0894, Q3=-0.0751, Q4=-0.0632

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=-0.0620, mid=-0.0466, high=-0.0320
- 流动性组: low=-0.0579, mid=-0.0432, high=-0.0612
- 换手率(5d top分位): 0.512

## 结论（基于 Universe_B）: **淘汰**

|RankICIR_5d|=0.26≤0.30
