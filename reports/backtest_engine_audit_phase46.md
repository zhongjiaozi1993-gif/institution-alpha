# 回测引擎审计 — Phase 4.6（timeline & portfolio accounting 修复）

审计时间: 2026-07-10
对象: [src/backtest/signal_backtester.py](../src/backtest/signal_backtester.py)、[src/backtest/metrics.py](../src/backtest/metrics.py)
测试: [tests/test_signal_backtester_timeline.py](../tests/test_signal_backtester_timeline.py)、[tests/test_signal_backtester_tradable_flags.py](../tests/test_signal_backtester_tradable_flags.py)

---

## 1. 本次修复的问题

| # | 问题（Phase 4.5 及以前） | 修复（Phase 4.6） |
|---|--------------------------|--------------------|
| 1 | 信号日 T 就把 T+1 仓位塞进 `open_positions`，mark-to-market 在 T 日用 `close_T / open_{T+1}` 混算，NAV 含未来仓位 | T 日只生成 `pending_order`；T+1 开盘才建仓（entry_date=T+1）；MTM 仅统计 entry_date ≤ today |
| 2 | 现金/成本口径混乱：入场按 `pos_weight*(1+cost)` 扣现金，退出又扣往返成本，滑点重复体现 | 份额制：买入 `cash-=slot_capital`（内含买手续费），卖出 `cash+=proceeds-sell_fee`；买/卖手续费、买/卖滑点各体现一次 |
| 3 | 买入只看 `buyable_flag` | 买入须同时 `buyable_flag=True` 且 `tradable_flag=True`（且非停牌） |
| 4 | 跌停/停牌不可卖时 `continue`，到期(==)判断使退出事件可能丢失 | 增加 `pending_exit`；触发 stop/take/maturity 但不可卖 → 记录原因，下一可卖日按**开盘价**卖出(deferred=True)；maturity 改为 `day_idx >= exit_idx` 不丢事件 |
| 5 | 策略模式隐式 | 明确 `strategy_mode="fixed_holding_fill_slots"`（固定持有期、有空槽且现金足够才补仓）；`daily_rebalance_topN` 后续再做 |
| 6 | `metrics.compute_full_metrics` 要求 `pnl/pnl_pct`，与引擎 `net_return_pct` 不兼容 | 支持 `net_return_pct`；区分 `portfolio_total_return`(NAV) 与 `trade_return_sum`(单笔求和) |

## 2. 组合会计口径（份额制，成本不重复扣）

```
slot_capital = initial_capital / max_positions        # 每槽固定投入(gross)
买入: buy_fee   = slot_capital × cost
      notional  = slot_capital − buy_fee
      shares    = notional / (open × (1+买滑点))
      cash     -= slot_capital
卖出: exec      = price × (1−卖滑点)
      proceeds  = shares × exec
      sell_fee  = proceeds × cost
      cash     += proceeds − sell_fee
NAV = cash + Σ(shares × 当日收盘)
单笔 net_return_pct = (proceeds − sell_fee) / slot_capital − 1
```

**成交价口径（已固定）**:
- 到期(maturity): 当日**收盘**卖出。
- 止损/止盈: 触发价(sl/tp)卖出。
- 顺延(deferred，跌停/停牌后): **下一可卖日开盘**卖出。

## 3. 测试覆盖（toy: ≤3 股 × 10 天，11 项全通过）

**timeline（5）**: T 信号仅 T+1 入场 / 信号日 NAV 不含未来仓位 / 买成本不重复扣(入场后 NAV=1−单次买费) / 买卖滑点各一次 / NAV 手工可复算。

**tradable_flags（6）**: 无 flags 基线建仓 / 涨停不可买 / tradable_flag=False 阻止买入 / 停牌入场日不建仓 / 跌停到期→pending_exit 次日开盘卖出(deferred) / 停牌到期不丢退出事件。

手工复算示例（cost=1%, 平价10, 1槽, 无滑点）:
入场后 NAV = 1 − 0.01 = **0.99**（若重复扣则 0.98）；到期后 NAV = **0.9801**；单笔净收益 **−1.99%**（往返 2×1%）。

## 4. 对真实回测的影响（口径修正 → 收益显著回落）

Signal027 × Universe_B × 5d × Top30，flags ON，SL−8%/TP+15%:

| 口径 | 年化 | Sharpe | 最大回撤 |
|------|------|--------|--------|
| Phase 4.5（旧会计，含未来仓位/重复扣费） | ~15.0% | ~0.52 | ~−20.8% |
| **Phase 4.6（修正）** | **5.2%** | **0.22** | **−27.8%** |

旧口径明显高估。修正后是可复现、可实盘对齐的真实数字（**本阶段不做收益优化**）。

## 5. 仍存在的已知简化（后续）

1. `fixed_holding_fill_slots` 用固定槽位资金（=初始/槽数），NAV 增长时不加仓，偏保守；`daily_rebalance_topN` 待实现。
2. deferred 卖出用「下一可卖日开盘」，若连续多日跌停则一直顺延，符合实盘但可能长期占用槽位。
3. ST 无名单数据源、无行业数据 → ST 剔除/行业集中度仍缺（见 `reports/backtest_engine_audit.md`）。
4. 停牌仅按 flags 判定，长期停牌整段缺行未用全市场日历补齐。
5. 尚未与 `ml_dataset`/`label` 宽表直接打通（当前 runner 从单信号 parquet 取数）。
