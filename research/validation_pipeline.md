# Validation Pipeline

> 统一的 Signal 验证流程。每个 Signal 进入 Candidate 状态前必须通过本 Pipeline。
> v1.0, 2026-07-09。

---

## 一、设计原则

1. **所有 Signal 同一套验证** — 不做特殊对待
2. **先样本内，再 OOS** — 禁止在同一个数据集上既训练又评估
3. **多维评估** — 不只看收益，还要看 IC、稳定性、一致性
4. **对标准备** — 每个 Signal 必须提供 Universe Benchmark 超额

---

## 二、验证流程

```
Signal.transform()
    ↓
1. Forward Return 计算 (1d/3d/5d/10d/20d)
    ↓
2. IC / RankIC 分析
    ↓
3. Group Return 分析 (分5组/10组)
    ↓
4. Win Rate 分析 (逐 horizon)
    ↓
5. Benchmark Excess 分析
    ├── Signal Internal Benchmark (信号间比较)
    ├── Universe Benchmark (同池全部股票等权)
    └── Index Benchmark (中证1000)
    ↓
6. 月度一致性检查
    ↓
7. Stability 计算
    ↓
8. 样本内决策: Pass → Candidate | Fail → Archived
    ↓ (如果 Pass)
9. OOT/OOS 验证
    ↓
10. OOS 决策: Pass → Production | Fail → Archived
```

---

## 三、验证指标

### 3.1 IC 分析

| 指标 | 计算 | 通过标准 |
|------|------|----------|
| IC Mean | corr(signal_T, fwd_Nd) 跨股票均值 | > 0.02 |
| IC Std | IC 的时序标准差 | — |
| ICIR | IC Mean / IC Std | > 0.3 |
| RankIC Mean | Spearman corr(signal_T, fwd_Nd) | > 0.03 |
| RankICIR | RankIC Mean / RankIC Std | > 0.4 |

### 3.2 收益分析

| 指标 | 计算 | 通过标准 |
|------|------|----------|
| Top Group Return | 信号最高组等权 forward return | > Universe mean |
| Bottom Group Return | 信号最低组等权 forward return | < Universe mean |
| Spread | Top - Bottom | > 0 |
| Win Rate (5d) | fwd_5d > 0 的比例 | > 52% |

### 3.3 超额分析

| 指标 | 计算 | 通过标准 |
|------|------|----------|
| Universe Excess (5d) | signal_fwd - universe_mean_fwd | > 0 |
| Universe Excess Win Rate | excess > 0 的比例 | > 50% |
| Index Excess (5d) | signal_fwd - 中证1000 fwd | > 0 (如果可用) |

### 3.4 一致性分析

| 指标 | 计算 | 通过标准 |
|------|------|----------|
| 月度正收益月数 | fwd_5d > 0 的月份数 | >= 8/12 |
| H1 vs H2 差异 | H2 fwd - H1 fwd | 不出现符号反转 |
| Stability | mean(universe_excess) / std(universe_excess) | > 0 |

### 3.5 数据质量

| 指标 | 通过标准 |
|------|----------|
| 价格缺失率 | < 5% |
| 前向收益缺失率 | < 10% |
| Top 5 贡献集中度 | < 50% |

---

## 四、验证输出格式

每个 Signal 完成验证后，输出:

```json
{
  "signal_id": "Signal004",
  "validation_date": "2026-07-09",
  "status": "Candidate",
  "in_sample_period": "2025-01 to 2025-12",
  "oos_period": null,
  "metrics": {
    "IC_mean": 0.035,
    "ICIR": 0.29,
    "RankIC_mean": 0.042,
    "RankICIR": 0.38,
    "group_return_spread_5d": 0.0295,
    "win_rate_5d": 0.595,
    "universe_excess_5d": 0.0114,
    "universe_excess_winrate_5d": 0.49,
    "monthly_positive_months": 10,
    "stability": 0.11,
    "data_quality": {
      "price_missing_pct": 0.0,
      "fwd_missing_pct": 0.025,
      "top5_concentration_pct": 0.333
    }
  },
  "candidate_universe": ["000547", "000657", "..."],
  "flags": ["in_sample_only", "no_oos_yet"]
}
```

---

## 五、OOT/OOS 分离规则

- **样本内 (IS)**: 2025-01 ~ 2025-12，用于初步筛选
- **样本外 (OOS)**: 2026-01 ~ 2026-06，用于最终确认

OOS 验证规则:
1. Signal 的 `fit()` 只能在 IS 上调用
2. `transform()` 可用在任意时段
3. OOS 的评估指标阈值与 IS 相同
4. OOS 未通过 → Signal 降级为 Archived
5. OOS 通过 → Signal 升级为 Production

---

## 六、与现有代码的映射

| Validation 步骤 | 现有代码 | 需调整 |
|-----------------|---------|--------|
| Forward Return | scale_oot_300.py: attach_forward_returns() | 抽取为独立模块 |
| Benchmark | scale_oot_300.py: universe_bench / signal_bench | 抽取为独立模块 |
| Per-stock Eval | scale_oot_300.py: evaluate_per_stock() | 改为跨 Signal 通用 |
| A/B/C/D 分组 | analyze_dbscan_stock_profile.py | 改为 validate() 输出的一部分 |
| 报告生成 | scale_oot_300.py 末尾 | 统一模板 |

---

## 七、实施优先级

1. **P0**: 抽取 `attach_forward_returns()` 和 benchmark 计算为独立模块 `src/validation/`
2. **P0**: 实现 `validate()` 方法（先用于 Signal004，验证与现有结果一致）
3. **P1**: 接入中证1000 index benchmark
4. **P1**: 对 Signal004 跑 2026 H1 OOS
5. **P2**: 新 Signal 接入时默认走本 Pipeline
