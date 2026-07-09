# Signal004: DBSCAN BUY — 定位回顾与重新评估

> 从"策略"调整为 Signal Zoo 中的一个 Signal。2026-07-09。

---

## 一、Signal 定义

| 字段 | 值 |
|------|-----|
| signal_id | Signal004 |
| signal_name | DBSCAN BUY |
| category | Microstructure |
| source | Self |
| data_requirement | Level2 逐笔成交（Wind GB18030 格式） |
| current_status | Candidate |
| validation_status | 样本内已验证（2025年全年） |

**信号逻辑:**

1. 对每只股票的 Level-2 逐笔成交数据，按日跑 DBSCAN 聚类（基于成交时间+价格+数量的空间密度）
2. 聚类结果中，方向为 BUY 的大单集群 → 产生 BUY 信号
3. 信号日 T，计算 T+N 日收盘价 vs T 日收盘价的 forward return

**关键参数:**

- DBSCAN eps / min_samples（V6 pipeline）
- 大单阈值: 10万股（普通大单）/ 50万股（超大单）
- 聚类附加字段: avg_price, avg_amount_wan, order_count, time_span_min, vwap_deviation_pct

---

## 二、数据需求

| 层级 | 内容 | 状态 |
|------|------|------|
| Level-2 逐笔成交 | Wind 格式 GB18030, 237天(2025), 300只股票 | 已有 |
| 日线 OHLCV | akshare Sina 来源 hfq, data/daily/{stock}.parquet | 已有 |
| 股票池 | DBSCAN 300 → Top100 按成交额 | 已有 |

**未解决的数据问题:**

- hfq/qfq 坐标不匹配导致 `entry_price` 收益不可用
- 仅覆盖 300 只 DBSCAN 输出股票，非全市场
- 行业/板块数据未接入

---

## 三、当前验证结果

### 3.1 总体评估

| 指标 | Priority 25 | Top100 | 全量 300 |
|------|------------|--------|----------|
| win_5d | **58.7%** | 48.7% | 46.1% |
| avg_fwd_5d | **+2.12%** | +0.53% | +0.11% |
| universe_excess_fwd_5d | — | **-0.83%** | — |
| 月度正收益 | 10/12 | 9/12 | — |

### 3.2 A/B/C/D 分组 (Top100)

| 分组 | 数量 | win_5d | excess_5d | 含义 |
|------|------|--------|-----------|------|
| A: 有效候选 | 13 | 59.5% | +1.14% | 信号有效 |
| B: 无超额 | 35 | — | 负 | 纯 Beta |
| C: 弱信号 | 8 | <=52% | — | 不可靠 |
| D: 无效 | 44 | 43% | -2.10% | 完全无效 |

### 3.3 有效股票特征

- **大单金额**: avg_amount_wan +58%（单笔 cluster 金额更大）
- **拆单委托数**: avg_order_count +46%（更多拆单 = 更像机构行为）
- **流动性**: avg_amount_yuan +76%（日均成交额更高）
- **非决定性**: 信号数量、聚类时长、VWAP 偏离

---

## 四、Candidate 股票池 (V0)

13 只（按 score 降序）:

000547（航天发展）、000657（中钨高新）、000426（兴业银锡）、000807（云铝股份）、
000572（海马汽车）、000510（新金路）、000887（中鼎股份）、000688（国城矿业）、
000859（国风新材）、000811（冰轮环境）、000617（中油资本）、000603（盛达资源）、
000833（粤桂股份）

**注意**: 这是研究候选池，不是实盘池。所有结论基于 2025 年样本内数据。

---

## 五、已知局限

1. **统计范围窄**: 300 只 DBSCAN 输出股票，不具全市场代表性
2. **无 OOS**: 全部基于 2025 年同一样本，未做训练/测试分离或 2026 年验证
3. **幸存者偏差风险**: Priority 25 的手工选择逻辑未文档化
4. **未评估交易成本**: 无滑点、无手续费、无冲击成本
5. **无市场状态适应**: V6 信号在 3/5/11 月失效原因未深入分析
6. **仅 BUY 方向**: SELL 信号未系统评估
7. **hfq 价格坐标**: entry_price 收益无法计算

---

## 六、下一步

### 继续做的

1. **2026 H1 OOS 验证**: 对 13 只 candidate 跑 out-of-sample
2. **SELL 信号评估**: 对称地跑 SELL 方向
3. **接入 qfq 价格**: 解锁 entry_price 收益口径
4. **行业特征补充**: 确认候选股票在行业维度是否有 pattern

### 不再做的

1. 不再扩 DBSCAN 股票池（Top100 → Top200 已否决）
2. 不再单独优化 V6（V6 作为 Signal004，不特殊对待）
3. 不将 V6 直接作为策略使用（需经 Validation → Model → Portfolio 完整流程）

### 在 Signal Zoo 中的位置

```
Signal004_DBSCAN_BUY
  └── 与 Price Signals (Alpha191) 组合 → Model 层
  └── 与 Order Flow Signals (OFI/Sweep) 组合 → Model 层
  └── 单独使用 → 不推荐（单一 Signal 不是策略）
```
