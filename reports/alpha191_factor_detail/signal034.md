# Signal034: Alpha191_VolPrice_GTJA102 — 跨 Universe 稳定性

窗口: 2025-01-01 ~ 2025-12-31  |  label: open-to-open(无未来函数)

## RankIC / RankICIR / spread（各 horizon × universe）

| Universe | 股票 | RankIC_1d | RankIC_5d | RankIC_10d | RankICIR_5d | spread_5d% | 扣费spread_5d% | 覆盖5d% |
|---|---|---|---|---|---|---|---|---|
| Universe_A | 300 | -0.0245 | -0.0377 | -0.0347 | -0.33 | -0.132 | -0.652 | 97.5 |
| Universe_B | 731 | -0.0180 | -0.0256 | -0.0248 | -0.27 | 0.053 | -0.467 | 97.5 |
| Universe_C | 160 | -0.0300 | -0.0453 | -0.0413 | -0.34 | -0.177 | -0.697 | 97.5 |

## 分季度 RankIC_5d（替代分年度）

- Universe_A: Q1=-0.0172, Q2=-0.0448, Q3=-0.0333, Q4=-0.0531
- Universe_B: Q1=+0.0026, Q2=-0.0349, Q3=-0.0242, Q4=-0.0416
- Universe_C: Q1=-0.0082, Q2=-0.0654, Q3=-0.0587, Q4=-0.0387

## 分市值组 / 分流动性组 RankIC_5d（Universe_B）

- 市值组: low=-0.0380, mid=-0.0274, high=-0.0116
- 流动性组: low=-0.0310, mid=-0.0261, high=-0.0269
- 换手率(5d top分位): 0.481

## 结论（基于 Universe_B）: **淘汰**

|RankICIR_5d|=0.27≤0.30
