# Level-2 信号归因与中性化报告（Phase 5.2C）

生成时间: 2026-07-12 14:38  |  purged OOS：train≤purge_cut / test≥2025-09-01  |  中性化控制: log成交额 + 换手 + log市值（行业无数据源，暂缺）

> purge: 5d 末train label结束日 2025-08-14 < 首test 2025-09-01；10d 2025-08-14 < 2025-09-01。全程 train 定方向、test 只评估。

---

## 1. 单特征基线 vs 35 综合分（purged OOS 定向 RankIC）

| 信号 | test RankIC(5d) | RankICIR(5d) | test RankIC(10d) | RankICIR(10d) |
|---|---|---|---|---|
| l2_amount_yi | +0.0493 | +0.25 | +0.0509 | +0.25 |
| l2_trade_count | +0.0352 | +0.25 | +0.0421 | +0.29 |
| l2_large_share | +0.0142 | +0.14 | +0.0098 | +0.10 |
| amount+volume 等权 | +0.0412 | +0.30 | +0.0462 | +0.32 |
| **35 特征综合分** | **+0.0438** | +0.26 | **+0.0458** | +0.29 |

> 35 综合分 vs 单一成交额(5d) 绝对 RankIC 差 = **-0.0055**。综合分相对成交额提升有限 → 主要信息来自成交额量级。

## 2. 特征族消融（各族综合分 purged OOS RankIC）

| 族 | 特征数 | test RankIC(5d) | RankICIR(5d) | test RankIC(10d) | RankICIR(10d) |
|---|---|---|---|---|---|
| flow | 7 | +0.0288 | +0.19 | +0.0370 | +0.23 |
| intraday+session | 5 | -0.0006 | -0.00 | +0.0257 | +0.22 |
| large | 12 | +0.0402 | +0.24 | +0.0409 | +0.23 |
| cluster | 11 | +0.0408 | +0.23 | +0.0381 | +0.21 |
| direction | 11 | +0.0022 | +0.02 | +0.0013 | +0.01 |
| **all(35)** | 35 | +0.0438 | +0.26 | +0.0458 | +0.29 |

## 3. 逐特征中性化（OOS 定向 RankIC_5d：原始 → 残差）

残差 = 每日截面对 [log成交额,换手,log市值] 回归后的余项。retained = |残差IC|/|原始IC|。

| 特征 | 族 | direction | 原始 IC | 中性化后 IC | retained |
|---|---|---|---|---|---|
| l2_amount_yi | flow |  | +0.0493 | +0.0246 | 50% |
| l2_super_buy_yi | large |  | +0.0446 | +0.0345 | 77% |
| l2_big_buy_yi | large |  | +0.0435 | +0.0219 | 50% |
| l2_cluster_buy_wan | cluster |  | +0.0431 | +0.0298 | 69% |
| l2_buy_cluster_count | cluster |  | +0.0405 | -0.0024 | 6% |
| l2_big_sell_yi | large |  | +0.0398 | +0.0255 | 64% |
| l2_max_cluster_wan | cluster |  | +0.0396 | +0.0367 | 93% |
| l2_super_sell_yi | large |  | +0.0389 | +0.0371 | 95% |
| l2_cluster_count | cluster |  | +0.0381 | +0.0011 | 3% |
| l2_cluster_sell_wan | cluster |  | +0.0374 | +0.0336 | 90% |
| l2_avg_cluster_orders | cluster |  | +0.0372 | +0.0312 | 84% |
| l2_cluster_buy_intensity | cluster | ✓ | +0.0364 | +0.0061 | 17% |
| l2_avg_order_wan | large |  | +0.0354 | -0.0038 | 11% |
| l2_trade_count | flow |  | +0.0352 | +0.0134 | 38% |
| l2_sell_cluster_count | cluster |  | +0.0349 | +0.0031 | 9% |
| l2_avg_trade_amt_wan | flow |  | +0.0331 | -0.0046 | 14% |
| l2_super_buy_ratio | large |  | +0.0300 | +0.0106 | 35% |
| l2_order_count | large |  | +0.0270 | +0.0193 | 72% |
| l2_volume_wan | flow |  | +0.0258 | +0.0163 | 63% |
| l2_early_net_ratio | session | ✓ | -0.0255 | +0.0318 | 125% |
| l2_vwap_close_dev | intraday |  | +0.0166 | +0.0069 | 42% |
| l2_active_sell_ratio | flow | ✓ | -0.0151 | -0.0290 | 192% |
| l2_net_active_ratio | flow | ✓ | -0.0150 | -0.0289 | 193% |
| l2_active_buy_ratio | flow | ✓ | -0.0150 | -0.0289 | 193% |
| l2_big_net_ratio | large | ✓ | +0.0149 | +0.0128 | 86% |
| l2_large_share | large |  | +0.0142 | +0.0143 | 100% |
| l2_big_net_yi | large | ✓ | -0.0142 | +0.0230 | 162% |
| l2_cluster_net_wan | cluster | ✓ | -0.0138 | +0.0133 | 96% |
| l2_buy_order_ratio | large | ✓ | +0.0126 | +0.0114 | 91% |
| l2_intraday_ret | intraday |  | +0.0100 | -0.0073 | 73% |
| l2_late_net_ratio | session | ✓ | -0.0095 | -0.0208 | 219% |
| l2_close_pos | intraday |  | +0.0082 | -0.0008 | 9% |
| l2_avg_cluster_hhi | cluster |  | -0.0075 | +0.0117 | 156% |
| l2_avg_cluster_vwap_dev | cluster |  | +0.0044 | +0.0015 | 34% |
| l2_super_net_yi | large | ✓ | +0.0005 | +0.0068 | 1262% |

> 中性化后 |RankIC_5d|>0.015 的特征：cluster **规模型 4/6**、cluster **计数/方向型 0/5**，direction 族 **6/11**。
> 注：`l2_active_buy_ratio/l2_active_sell_ratio/l2_net_active_ratio` 三者 IC 完全相同（同一净主买信号）；retained>100% 多为原始 IC≈0 时的比值放大，绝对量仍在噪声级(|IC|<0.035)。

## 4. 关键问答

**Q1 综合分 vs 成交额**：见 §1，35 综合分(5d) +0.0438 vs 成交额 +0.0493。

**Q2 cluster 残余 IC**：中性化后 cluster **规模型**留存 4/6 只（max_cluster_wan/cluster_sell_wan 等，本质是线性 log成交额未吸干净的**残余规模**），而 **计数/方向型**（buy/sell_cluster_count、cluster_count、cluster_buy_intensity、cluster_net_wan）仅 0/5 留存（retained 多为 3~17%，基本塌缩）→ **cluster 的残余是规模而非拆单方向**，不支持独立机构指纹。

**Q3 direction 分市值组（5d，test 窗口）**：

| 市值组 | direction综合 RankIC | RankICIR | 有效日 |
|---|---|---|---|
| small | -0.0176 | -0.15 | 76 |
| mid | +0.0009 | +0.01 | 76 |
| large | -0.0078 | -0.06 | 76 |

> direction 综合分各市值组 |IC| 均弱(<0.02)，方向假设仍未被验证。

**Q4 10d 增量是否规模重复**：35 综合分(10d) test RankIC 原始 +0.0458 → 中性化(去 log成交额/换手/log市值) 残差 **-0.0069**。残差大幅塌缩 → 10d 增量**主要是规模因子的重复表达**。

## 5. 门槛结论

> **中性化后 cluster/direction 基本归零，综合分主要是规模/流动性代理** → Level-2 当前定位为**流动性/规模增强模块**，Phase 6 可小权重使用，**但不称为机构 Alpha**；机构拆单方向识别需换特征/换标签或更细粒度，暂不投入 V6 扩展。

## 6. 已知限制

1. 中性化控制缺**行业**（无数据源），残余 IC 可能仍含行业暴露。
2. 单一 train/test 切分；direction 分组样本随市值分层进一步变薄。
3. 中性化为逐日线性 OLS，未做稳健回归/非线性控制。
