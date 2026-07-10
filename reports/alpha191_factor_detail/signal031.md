# Signal031: Alpha191_VolPrice_GTJA011 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | -0.0271 | -0.0262 | -0.0194 | -0.22 | -0.296 | -0.816 | 97.5 |
| Universe_B | 731 | -0.0244 | -0.0236 | -0.0196 | -0.23 | -0.221 | -0.741 | 97.5 |
| Universe_C | 160 | -0.0244 | -0.0212 | -0.0203 | -0.15 | -0.097 | -0.617 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.0190, Q2=-0.0373, Q3=-0.0303, Q4=-0.0157
- Universe_B: Q1=-0.0044, Q2=-0.0263, Q3=-0.0415, Q4=-0.0173
- Universe_C: Q1=-0.0271, Q2=-0.0268, Q3=-0.0123, Q4=-0.0200

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=-0.0196, mid=-0.0245, high=-0.0215
- 流动性组: low=-0.0057, mid=-0.0246, high=-0.0287
- 换手率(5d top分位): 0.311

## 结论（基于 Universe_B）: **淘汰**

|RankICIR_5d|=0.23≤0.30
