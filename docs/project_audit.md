# 项目审计报告（Phase 0）

生成时间: 2026-07-10
范围: `institution-alpha` 全仓库
目的: 在升级为「ML-ready 日频量化研究平台」前，先摸清现有资产、口径与风险。

---

## 1. 当前目录结构（2 层）

```
institution-alpha/
├── config/                 # 单数，含 settings.yaml（7 段配置）
├── configs/                # 新增（本轮），存放 universe.yaml 等
├── data/
│   ├── daily/              # 1222 个 parquet（个股+指数日线，2025 全年 243 交易日）
│   ├── daily_bak_wrong_unit/  # 565 个错误单位备份（待清理）
│   ├── lhb/                # 龙虎榜缓存
│   ├── level2/             # 空目录
│   ├── single_stock/       # 224 个个股子目录（Level-2 衍生数据来源）
│   └── processed/          # 画像/信号/验证/股票池等产物（447 MB）
├── docs/                   # 新增（本轮）
├── models/                 # 已有 LightGBM 产物
├── reports/                # 新增（本轮）
├── research/               # 4 篇调研文档（含 alpha191_survey.md）
├── scripts/                # 42 个 .py + windows/（31 个 .ps1/.bat）
├── signal_zoo/
│   ├── external/alpha191/  # Alpha191 适配器 + 30 因子清单
│   ├── registry/           # universe_registry.csv + signal_registry.csv
│   └── categories/         # 4 个空骨架目录（仅 .gitkeep）
├── src/                    # 9 个子模块（见 §2）+ 本轮新增 4 个
└── tests/
```

根目录还散落 6 个 .py（`build_behavior_dataset.py` 等），建议后续移入 `scripts/`。

## 2. 已有脚本 / 模块说明

**`src/` 现有子模块：**

| 模块 | 关键文件 | 职责 |
|------|----------|------|
| `src/data/` | price_loader / lhb_collector / level2_reader | 数据采集与缓存 |
| `src/cluster/` | split_detector / institution_tracker / behavior_db | Level-2 拆单聚类 |
| `src/alpha/` | alpha_profiler / dynamic_scorer / return_calculator | Alpha 归因 |
| `src/signal/` | generator / composite / decay | 信号生成/合成/衰减 |
| `src/backtest/` | signal_backtester / engine / metrics | 回测（两套引擎，见 §7） |
| `src/risk/` | regime / crowding / decay_monitor | 风控 |
| `src/evidence/` | chain / behavior / public_sources / market | 单票证据链 |
| `src/pipeline/` | daily_runner | 日频调度 |

**本轮新增：** `src/registry/`、`src/features/`、`src/dataset/`、`src/validation/`。

**核心脚本（保留）：** `validate_signal_daily.py`、`batch_validate_signals.py`（验证框架）；`sofia_*`（Level-2）；`institution_*`（机构识别）；`build_evidence_chain.py`。

**一次性/可清理脚本：** `batch_run_v4_v6*.py`、`robustness_validation.py`、`run_unified_robustness_grid.py`、`scale_oot_300.py`、`validate_price_units.py`、`check_*` 等（产物已落地，代码可归档）。

## 3. 已有数据表说明

| 数据 | 位置 | 规模 | 口径要点 |
|------|------|------|----------|
| 个股日线 | `data/daily/{code}.parquet` | 1222 文件 / 2025 全年 | 列: date,open,high,low,close,volume,amount,outstanding_share,turnover。**close 为后复权(hfq)**，非真实成交价 |
| 中证1000 指数 | `data/daily/idx_000852.parquet` | 2843 行 | 列: date,open,high,low,close,volume。close 6000+ 点，**口径正确** |
| 中证1000（错误） | `data/daily/000852.parquet` | 243 行 | 价格 ~38-44，**单位错误，勿用** |
| Alpha191 因子 | `data/processed/signals/price_alpha191_full/` | 30 个 parquet | 覆盖 776 股 × 243 日，Signal017-046 |
| 龙虎榜 | `data/lhb/` | 27 文件 | 席位级明细 |
| Level-2 衍生 | `data/single_stock/{code}/` | 224 个股 | 176 个有对应日线 |

**关键：** 全库数据窗口 = 2025 全年（243 交易日）。**只有单一年份**，无法做严格的分年度/多市场阶段验证；本轮以分季度/分月替代，并明确标注该限制。

## 4. Signal Registry 格式

文件: `signal_zoo/registry/signal_registry.csv`（46 行，Signal001-046）
列: `signal_id, signal_name, category, source, source_library, source_formula_id, data_requirement, frequency, status, validation_status, recommended_direction, notes, default_universe_id`

**缺失字段: `available_time`（未来函数防护的关键字段，本轮需补）。**

## 5. Universe Registry 格式

文件: `signal_zoo/registry/universe_registry.csv`
列: `universe_id, universe_name, data_requirement, source, stock_count, start_date, end_date, status, notes`
现有 2 条: Universe001(292 Level-2) / Universe002(782 日线)。命名不表意，本轮重构为 Universe_A/B/C（见 universe_report.md）。**无 `universe_registry.py`，仅 CSV。**

## 6. Validation Pipeline 输入/输出

**输入（信号 parquet）：** `trade_date, stock_code, signal_value`（+ 可选 signal_id/name/source）。
**远期收益：** `precompute_forward_returns()` 生成 `fwd_{1,3,5,10,20}d` 与 `win_{h}d`。
**输出：** 每信号 Markdown 报告 + summary CSV（IC/RankIC/RankICIR/spread/三类胜率/方向分类）；批量版额外产出 candidate 清单。

**⚠️ 口径问题：** `precompute_forward_returns` 用 **收盘价 close[t+h]/close[t]** 计算远期收益，基点对齐信号 `trade_date`。若信号于 T 日盘后可得，真实入场应为 **T+1 开盘**，二者存在隔夜跳空差异，会高估收益。本轮 `label_builder` 改为 **T+1 open → T+(1+h) open**，从源头消除该未来函数。

## 7. Backtest Engine 交易假设

主引擎 `src/backtest/signal_backtester.py`（`engine.py` 为冗余旧版，建议删除）。

| 约束 | 是否实现 | 说明 |
|------|----------|------|
| T+1 入场 | ✅ | 用 T 日信号，T+1 开盘价入场 |
| 手续费 | ✅ | cost_bps 默认 20 |
| 滑点 | ✅ | slippage_bps 默认 10 |
| 止损/止盈 | ✅ | 可配 |
| 持仓数上限 | ✅ | max_positions |
| 冷却期 / 趋势过滤 | ✅ | cooldown_days / ma20-ma60 |
| **涨停禁买** | ❌ | 无检查 |
| **跌停禁卖** | ❌ | 无检查 |
| **停牌处理** | ⚠️ | 仅当价格缺失时静默跳过，非显式 |
| **ST / 新股 / 低流动性剔除** | ❌ | 无 |
| **行业集中度** | ❌ | 无 |

## 8. 潜在未来函数

1. **验证远期收益口径**（§6）：close-to-close 对齐信号日 → 隐含 T 日隔夜收益。**本轮已修**（open-to-open）。
2. 回测引擎入场（signal_backtester）：T+1 开盘，**无未来函数**。旧 `engine.py` 未用。
3. 因子本身：Alpha191 由个股 OHLCV 滚动计算，T 日收盘后可得，标注 `available_time = T 收盘后`，只可用于 T+1，**无未来函数**（前提是 signal 的 trade_date=计算日）。

## 9. 交易约束覆盖现状

| 项 | 现状 |
|----|------|
| 涨跌停 | ❌ 回测未处理；本轮 `tradable_flag_builder` 生成 limit_up/limit_down_flag |
| 停牌 | ⚠️ 隐式；本轮以 volume==0 显式标记 suspend_flag |
| ST | ❌ **无 ST 名单数据源**，本轮 st_flag 默认 False 并标注为已知缺口 |
| 新股 | ❌ 本轮以「上市/有数据交易日 < 60」近似 new_stock_flag |
| 低流动性 | ❌ 本轮以近 20 日均额阈值生成 low_liquidity_flag |
| 滑点/手续费 | ✅ 回测已含 |
| T+1 | ✅ 回测已含；label 亦按 T+1 open 定义 |

## 10. 最需要重构的地方（优先级）

1. **统一远期收益/label 口径**（消除 close-to-close 未来函数）→ 本轮 `label_builder`。
2. **补 tradable_flag**（涨跌停/停牌/ST/新股/流动性）→ 本轮 `tradable_flag_builder`。
3. **Universe 重构**（表意命名 A/B/C，扩池）→ 本轮 `universe_registry`。
4. **market cap 口径**：不可用 hfq close×股本（会得 32 万亿的荒谬值）；须用真实价≈`amount/volume` × `outstanding_share`。
5. **指数文件去重**：删 `000852.parquet`（错误），统一用 `idx_000852.parquet`。
6. **回测引擎去冗余**：删 `src/backtest/engine.py`。
7. **signal_registry 补 `available_time` / `version`**。
8. **ST/行业数据源缺失**：需接入 ST 名单与行业分类，否则相关剔除与分组无法严格执行。

---

## 已知数据缺口（影响本轮范围）

- **无行业分类映射** → 分行业 RankIC、行业超额 label、行业集中度约束暂不可用（标注 N/A）。
- **无 ST/*ST 名单** → st_flag 只能占位。
- **单一年份（2025）** → 分年度验证降级为分季度/分月。
- **无中证500 指数文件**（idx_000905 缺失）→ 超额 label 仅对中证1000。
