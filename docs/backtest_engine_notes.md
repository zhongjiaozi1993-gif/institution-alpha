# 回测引擎交易假设说明（signal_backtester）

对应代码: [src/backtest/signal_backtester.py](../src/backtest/signal_backtester.py)
标准入口: [scripts/run_signal_backtest.py](../scripts/run_signal_backtest.py)
更新: 2026-07-10（Phase 4.5 接入 tradable_flags）

---

## 1. 时序与撮合口径（无未来函数）

| 环节 | 口径 |
|------|------|
| 信号产生 | T 日收盘后，仅用 T 及之前信息 |
| 入场 | **T+1 开盘价**（+滑点） |
| 到期退出 | T+1+N 日收盘价（N = `holding_days`） |
| 止损/止盈 | 持有期内盘中触发（用当日 high/low 判定，按预设价成交） |

入场严格用 T+1 open，回测引擎本身**无未来函数**（审计见 `docs/project_audit.md` §8）。

## 2. 交易约束（Phase 4.5 接入 tradable_flags）

引擎 `run(signals, prices, tradable_flags=...)` 接入
`data/processed/tradable/tradable_flags.parquet`：

| 约束 | 规则 | 实现 |
|------|------|------|
| 涨停不可买 | T+1 `buyable_flag=False`（涨停/停牌）→ 放弃该信号 | `not_buyable` 集合 |
| 跌停不可卖 | 退出日 `sellable_flag=False`（跌停/停牌）→ 卖出顺延到下一可卖日 | `not_sellable` 集合 |
| 停牌不可交易 | `suspend_flag=True` → 当日既不买也不卖，持仓顺延 | `suspended` 集合 |

`tradable_flags=None` 时退化为无约束（旧行为），便于对照。命令行 `--no-flags` 可关闭。

**实测影响**（Signal027 × Universe_B × 5d × Top30）：
约束 ON vs OFF，年化 15.0% vs 22.3%，Sharpe 0.52 vs 0.74。
差异来自「买不进涨停、卖不掉跌停」——这是真实可实现收益，不是退化 bug。

## 3. 成本模型

| 项 | 默认 | 说明 |
|----|------|------|
| 佣金 `cost_bps` | 20 bps | 入场扣一次，退出扣一次 |
| 滑点 `slippage_bps` | 10 bps | 入场价上浮，退出成本再计一次 |
| 单笔净收益 | `gross - (cost+slippage)*2/1e4` | 往返成本 |

## 4. 组合与权重

- 等权：每仓位权重 = `1 / max_positions`，无杠杆。
- `max_positions` 同时限制持仓数与单票权重上限。
- 现金归一化（初始 1.0），退出后资金可复用。
- **NAV 口径**用于最大回撤/年化/Sharpe（见 `metrics.compute_full_metrics`）。
  引擎 summary 里的 `total_return` 是**单笔净收益求和**（活跃度度量，非组合复利收益），勿混淆。

## 5. 其他可选控制

- `stop_loss` / `take_profit`：小数（如 -0.08 / 0.15）。
- `cooldown_days`：退出后同股冷却。
- `stock_trend_filter`：`ma20`/`ma60`，仅用 T 及之前数据（无未来函数）。

## 6. 已知简化（待后续完善）

1. 退出被跌停/停牌顺延时，成交价仍用原触发价（sl/tp）或当日 close，未精确到"下一可卖日实际价"——偏差有界，已记录。
2. 停牌仅以 flags 的 `suspend_flag`（volume==0）判定，长期停牌整段缺行未通过全市场日历补齐。
3. ST 无名单数据源，`st_flag` 恒 False，不影响撮合但影响纳池。
4. 绝对收益量级依赖等权/资金复用模型，**本阶段不做收益优化**，只保口径正确。
