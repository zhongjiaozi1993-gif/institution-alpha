# Signal018: Alpha191_VolumePrice_GTJA004 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | 0.0060 | 0.0055 | 0.0018 | 0.06 | NA | NA | 97.5 |
| Universe_B | 731 | 0.0051 | 0.0080 | 0.0073 | 0.11 | NA | NA | 97.5 |
| Universe_C | 160 | 0.0042 | 0.0127 | 0.0142 | 0.13 | NA | NA | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=+0.0203, Q2=-0.0140, Q3=+0.0072, Q4=+0.0112
- Universe_B: Q1=+0.0129, Q2=-0.0113, Q3=+0.0144, Q4=+0.0171
- Universe_C: Q1=+0.0384, Q2=-0.0038, Q3=+0.0082, Q4=+0.0127

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=+0.0041, mid=+0.0080, high=+0.0133
- 流动性组: low=+0.0079, mid=+0.0094, high=+0.0051
- 换手率(5d top分位): NA

## 结论（基于 Universe_B）: **淘汰**

|RankIC_5d|=0.008≤0.015; |RankICIR_5d|=0.11≤0.30; 扣费后 spread 方向不利(+nan%)
