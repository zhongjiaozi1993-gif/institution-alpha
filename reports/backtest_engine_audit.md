# 回测引擎交易约束审计（signal_backtester）

审计时间: 2026-07-10（Phase 4.5）
对象: [src/backtest/signal_backtester.py](../src/backtest/signal_backtester.py)

对照 A 股实盘约束清单，逐项标注支持状态。✅=已支持，⚠️=部分/近似，❌=未支持。

---

## 已支持 / 部分支持 / 未支持

| 约束 | 状态 | 实现位置 / 说明 |
|------|------|------|
| T+1 交易 | ✅ | 信号 T 日，入场 `all_dates[day_idx+1]` 开盘价 |
| 涨停不可买 | ✅ | Phase 4.5：T+1 `buyable_flag=False` → 放弃信号 |
| 跌停不可卖 | ✅ | Phase 4.5：退出日 `sellable_flag=False` → 顺延下一可卖日 |
| 停牌不可交易 | ✅ | Phase 4.5：`suspend_flag=True` → 当日不买不卖 |
| 手续费 | ✅ | `cost_bps` 入场/退出各一次 |
| 滑点 | ✅ | `slippage_bps` 入场价上浮 + 退出成本 |
| 止损 / 止盈 | ✅ | `stop_loss` / `take_profit`，盘中 high/low 判定 |
| 持仓数量上限 | ✅ | `max_positions` |
| 单票权重上限 | ✅ | 等权 = 1/`max_positions`（隐含上限） |
| 冷却期 | ✅ | `cooldown_days` |
| 趋势过滤 | ✅ | `stock_trend_filter` ma20/ma60，仅用 T 及之前 |
| 新股剔除 | ⚠️ | 引擎不判定；靠 universe/`tradable_flag` 上游过滤（new_stock_flag） |
| 低流动性剔除 | ⚠️ | 同上，上游 `low_liquidity_flag` |
| ST 剔除 | ❌ | 无 ST 名单数据源，`st_flag` 恒 False |
| 长期停牌整段缺行 | ⚠️ | 仅按 volume==0 判停牌，未用全市场日历补齐缺失交易日 |
| 行业集中度上限 | ❌ | 无行业映射数据 |
| 未来函数 | ✅ 无 | 入场 T+1 open；趋势过滤仅用 T 及之前 |

## 与 tradable_flags 的接线

- 引擎 `run(..., tradable_flags=df)` 接收 `data/processed/tradable/tradable_flags.parquet`。
- 三个 O(1) 查询集合：`not_buyable` / `not_sellable` / `suspended`（键 `(symbol, 'YYYY-MM-DD')`）。
- 键对齐：signals/prices/flags 的 symbol 统一 `zfill(6)`，日期统一 `%Y-%m-%d`。

## 实测：约束是否真正 binding

Signal027 × Universe_B × 5d × Top30（2025 全年）:

| 口径 | 年化 | Sharpe | 最大回撤(NAV) | 胜率 |
|------|------|--------|--------|------|
| flags ON（真实） | 15.0% | 0.52 | -20.8% | 46.9% |
| flags OFF（对照） | 22.3% | 0.74 | -19.0% | 47.3% |

约束使年化下降 ~7pp、Sharpe 下降 ~0.22——因买不进涨停、卖不掉跌停。约束确实生效。

## 后续待办（不属本次范围）

1. ST 名单、行业分类两个数据源接入后，补 ST 剔除与行业集中度约束。
2. 全市场交易日历补齐长期停牌缺行。
3. 顺延卖出改为「下一可卖日实际成交价」，替代当前近似（原触发价/当日 close）。
4. 与 ml_dataset / label 打通：直接从宽表取信号与 label，统一回测入口。
