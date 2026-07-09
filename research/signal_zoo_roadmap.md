# Signal Zoo Roadmap

> 规划 Signals 的优先级、来源和实施路径。v1.0, 2026-07-09。

---

## 一、当前状态

| 类别 | 已注册 | 已实现 | Candidate | Production |
|------|--------|--------|-----------|------------|
| Price | 7 | 0 | 0 | 0 |
| Order Flow | 6 | 0 | 0 | 0 |
| Microstructure | 3 | 1 | 1 | 0 |
| Fundamental | 3 | 0 | 0 | 0 |
| **合计** | **20** | **1** | **1** | **0** |

已实现: Signal004_DBSCAN_BUY（Candidate 状态）

---

## 二、Phase 1: 建立基础设施 (当前 Sprint)

**目标**: 完成架构，不新增代码实现。

- [x] Signal Registry + Schema
- [x] 目录结构
- [x] Alpha191 调研
- [x] DBSCAN 定位回顾
- [x] Validation Pipeline 设计
- [ ] Validation 模块抽取 (P0, 下个 Sprint)

---

## 三、Phase 2: Price Signals 接入 (Sprint 2)

**目标**: 接入 4 个 Alpha191 Price 因子，打通从外部库到 Validation 的完整链路。

| Signal ID | 名称 | 来源 | 优先级 | 理由 |
|-----------|------|------|--------|------|
| Signal017 | Alpha191_K_UP_FRAC | aurumq-rl #16 | P0 | 日内偏度，直观易懂 |
| Signal018 | Alpha191_TURN_STD | aurumq-rl #56 | P0 | 换手率波动，A股有效 |
| Signal019 | Alpha191_VOL_CORR | aurumq-rl #85 | P1 | 量价相关，经典 |
| Signal020 | Alpha191_HL_SPREAD | aurumq-rl #127 | P1 | 高低价差动量 |

**实施步骤:**

1. Clone aurumq-rl, 提取 `factor/` 核心模块
2. 适配数据格式（daily parquet → Polars DataFrame）
3. 为 4 个因子实现 Signal 接口
4. 在 Top100 universe 上跑 Validation Pipeline
5. 通过验证的 → Candidate; 未通过 → Archived

---

## 四、Phase 3: Order Flow Signals (Sprint 3)

**目标**: 基于已有 Level-2 数据开发 2-3 个订单流 Signal。

| Signal ID | 名称 | 来源 | 优先级 | 理由 |
|-----------|------|------|--------|------|
| Signal007 | OFI_Basic | Self | P0 | 订单流不平衡，L2最基础信号 |
| Signal009 | Cancel_Ratio | Self | P1 | 撤单率，A股有特色 |
| Signal010 | Aggressive_Buy_Ratio | Self | P1 | 激进买入检测 |

**实施步骤:**

1. 从 Level-2 逐笔数据计算 OFI
2. 日聚合为 Signal 输出
3. 在 Top100 universe 上跑 Validation
4. 与 Signal004 (DBSCAN BUY) 做相关性检查

---

## 五、Phase 4: 组合与 Model 层 (Sprint 4)

**目标**: 设计 Signal 组合方案，不接 ML。

| 步骤 | 内容 |
|------|------|
| Signal 相关性矩阵 | 检查同类别 Signal 是否冗余 |
| 简单组合 | 等权组合、IC加权组合 |
| 股票级信号聚合 | 多 Signal → 单股票综合评分 |
| 组合评估 | 组合后的 win_rate, excess, stability |

---

## 六、远期规划 (Sprint 5+, 暂不细化)

| 方向 | 内容 |
|------|------|
| 新 Price Signals | Alpha191 剩余因子选择性接入 |
| L2 深度 | Queue Imbalance, Sweep, Iceberg |
| 基本面 | PE/PB/ROE 基础版 |
| Model 层 | 轻量 ML（线性模型优先，不接 Transformer） |
| 行业中性 | 行业分类数据接入后的中性化处理 |

---

## 七、不做的事情

- 不一次性实现全部 Alpha191（先选 4 个验证流程）
- 不自己重新实现 Alpha191（用 aurumq-rl）
- 不接 LightGBM/XGBoost/Transformer（Phase 4 之前）
- 不扩 DBSCAN 股票池
- 不单独优化 V6

---

## 八、Sprint 对应

| Sprint | 内容 | 预计产出 |
|--------|------|----------|
| Sprint 1 (当前) | 架构 + 调研 + 设计 | Registry, Schema, Survey, Review, Pipeline, Roadmap |
| Sprint 2 | Price Signals 接入 | 4 个 Alpha191 Signal + Validation |
| Sprint 3 | Order Flow Signals | 2-3 个 OF Signal + Validation |
| Sprint 4 | 组合 + Model 设计 | 相关性矩阵 + 简单组合评估 |
| Sprint 5+ | 待定（根据 Phase 2-4 结果调整） | — |
