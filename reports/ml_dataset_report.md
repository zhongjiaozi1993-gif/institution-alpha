# ML-ready Dataset 报告

生成时间: 2026-07-10  |  窗口: 2025-01-01 ~ 2025-12-31

> 宽表格式: trade_date | symbol | in_Universe_* | feat_* | label_* | *_flag
> feature 于 T 收盘后可得(available_time=T_close)，label 于 T+1 open 起算，无未来函数。
> 本版 feature 仅含 Alpha191；Level-2/龙虎榜/北向 将按相同键增量并入。

---

## 概览

- 形状: **191913 行 × 52 列**
- 日期范围: 2025-01-02 ~ 2025-12-31（243 日）
- 股票数: 791
- feature 数: 30  |  label 数: 8  |  flag 数: 9

## Universe 分布

| Universe | 行数 | 股票数 |
|---|---|---|
| Universe_A | 72802 | 300 |
| Universe_B | 177338 | 731 |
| Universe_C | 42484 | 175 |

## Feature 缺失率（Top/Bottom 5）

| feature | 缺失率 |
|---|---|
| feat_signal041 | 1.9% |
| feat_signal036 | 1.9% |
| feat_signal037 | 2.3% |
| feat_signal017 | 2.4% |
| feat_signal019 | 3.9% |
| feat_signal045 | 11.2% |
| feat_signal027 | 11.2% |
| feat_signal043 | 16.0% |
| feat_signal020 | 17.3% |
| feat_signal032 | 69.1% |

> feature 整体平均缺失率: 9.3%（部分 Level-2 股票无 Alpha191 覆盖）

## Label 缺失率

| label | 缺失率 |
|---|---|
| label_1d | 0.8% |
| label_3d | 1.6% |
| label_5d | 2.5% |
| label_10d | 4.5% |
| label_1d_excess_index | 0.8% |
| label_3d_excess_index | 1.6% |
| label_5d_excess_index | 2.5% |
| label_10d_excess_index | 4.5% |

## 按年份分布

| 年份 | 行数 |
|---|---|
| 2025 | 191913 |

> 全库仅 2025 单一年份，跨年验证暂不可行。

## 未来函数自检

- feature 全部来自 T 日及之前的 OHLCV，available_time=T_close。
- label 全部来自 T+1 open 及之后，仅用于验证/训练，未混入 feature。
- feature 与 label 同键对齐于 T，feature 可用时点严格早于 label，**无泄漏**。
