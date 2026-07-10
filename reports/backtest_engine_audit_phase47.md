# 回测引擎审计 — Phase 4.7（edge-case cleanup）

审计时间: 2026-07-10
对象: [src/backtest/signal_backtester.py](../src/backtest/signal_backtester.py)、[src/backtest/metrics.py](../src/backtest/metrics.py)、[scripts/run_signal_backtest.py](../scripts/run_signal_backtest.py)
测试: [tests/test_signal_backtester_timeline.py](../tests/test_signal_backtester_timeline.py)、[tests/test_signal_backtester_tradable_flags.py](../tests/test_signal_backtester_tradable_flags.py)

本阶段只做边界修正与口径统一，**不改变会计逻辑、不做收益优化**。真实回测数字与 Phase 4.6 一致（NAV +4.98% / 年化 5.2% / Sharpe 0.22 / 最大回撤 −27.8%）。

---

## 1. 本次修复的问题

| # | 问题（Phase 4.6 及以前） | 修复（Phase 4.7） |
|---|--------------------------|--------------------|
| 1 | **停牌到期退出丢 deferred**：到期日恰逢停牌时，Step1 开头 `if suspended: continue` 直接跳过，未登记 `pending_exit`。下一可卖日走普通路径按**收盘价**卖出、`deferred=False`，与「跌停顺延」口径不一致 | 停牌当日若 `day_idx >= exit_idx` 且未挂起，先置 `pos.pending_exit_reason="maturity"` 再 continue；下一可卖日按**开盘价**成交、`deferred=True` |
| 2 | `stock_code` 补零只在入口脚本做，引擎内部直接 `astype(str)`，非补零输入（int `1`/`"1"`）与 6 位 `prices`/`flags` 键不匹配 | 引擎 `run()` 内部统一 `astype(str).str.zfill(6)`，不再只依赖入口脚本 |
| 3 | `summary.total_return` 实为 `trade_return_sum`（单笔求和），与 NAV 口径同名混淆 | 删除 `summary.total_return`；单笔求和只叫 `trade_return_sum`；组合收益只叫 `portfolio_total_return`（NAV）。`metrics.py` 也删除 `total_return` 别名 |
| 4 | `run_signal_backtest.py` 用 `compute_full_metrics(equity, 空DataFrame)`，丢弃交易口径指标；且打印引用已删除的 `total_return` | 改为 `compute_full_metrics(equity, trades)`；打印 `portfolio_total_return`、`trade_return_sum`、`deferred_exits` 及期末未平仓 |
| 5 | 无期末未平仓统计，长期停牌/连续跌停顺延卡住的仓位在报表中不可见 | summary 新增 `open_positions_at_end`、`unrealized_position_value`、`unrealized_pnl_pct`、`unrealized_nav_contribution` |

## 2. 停牌到期退出：修复前后对比

场景：d3 入场（holding=1）→ d4 到期，但 d4 停牌不可交易 → d5 恢复交易（d5 开盘价 9 ≠ 收盘价 11）。

| | 修复前（4.6） | 修复后（4.7） |
|---|---|---|
| d4（停牌+到期） | `continue`，什么都不记 | 登记 `pending_exit_reason="maturity"` |
| d5（可卖） | 普通路径 → maturity → **收盘 11** 卖出，`deferred=False` | pending 路径 → **开盘 9** 卖出，`deferred=True` |
| exit_date | 2025-01-06 | 2025-01-06 |

与「跌停顺延」完全一致：**顺延退出统一按下一可卖日开盘价成交，标记 deferred**。
由 `test_suspend_maturity_sets_pending_exit_and_sells_next_open` 用 open≠close 专门锁定成交价来源。

## 3. summary 字段口径（统一后）

```
portfolio_total_return   组合 NAV 首末比（权威组合收益，小数，如 0.0498=+4.98%）
trade_return_sum         单笔 net_return_pct 求和（活跃度，非组合口径，百分数，如 149.54）
open_positions_at_end    期末仍未平仓的仓位数
unrealized_position_value 期末未平仓市值 = Σ(shares × 末日收盘)
unrealized_pnl_pct       (未平仓市值 / Σgross_alloc − 1) × 100
unrealized_nav_contribution 未平仓市值 / 末日 NAV
```

- `total_return` 已从 summary 与 metrics 中**删除**（不再存在同名歧义字段）。
- `portfolio_total_return`（NAV 口径，0.2）与 `trade_return_sum`（单笔求和，20.0）数值/量纲均不同，由 `test_summary_total_return_is_portfolio_not_trade_sum` 用同一笔 +20% 交易验证两者分离。

## 4. 测试覆盖（toy: ≤3 股 × 10 天，15 项全通过）

**timeline（8）**: 原 5 项 + 新增 3 项：
- `test_stock_code_zfill_inside_engine`：signal 传 int `1` → 引擎 zfill 后匹配 `prices("000001")` 建仓。
- `test_summary_total_return_is_portfolio_not_trade_sum`：summary 无 `total_return`；`portfolio_total_return=0.2` 与 `trade_return_sum=20.0` 分离。
- `test_metrics_accepts_trades_schema`：`compute_full_metrics(equity, trades)` 接受 `net_return_pct` schema，返回组合+交易两口径且不崩。

**tradable_flags（7）**: 原 6 项 + 新增 1 项：
- `test_suspend_maturity_sets_pending_exit_and_sells_next_open`：停牌到期 → pending_exit=maturity → 下一可卖日**开盘价**成交、deferred=True。

## 5. 对真实回测的影响

口径未变，仅边界与报表修正。Signal027 × Universe_B × 5d × Top30，flags ON，SL−8%/TP+15%:

| 指标 | 值 |
|------|-----|
| 组合总收益(NAV) | +4.98% |
| 年化 | 5.2% |
| Sharpe / Sortino / Calmar | 0.22 / 0.25 / 0.19 |
| 最大回撤(NAV) | −27.8% |
| 交易数 / 胜率 | 1376 / 44.2% |
| 止损 / 止盈 / 顺延退出 | 236 / 150 / 32 |
| 期末未平仓 | 0 笔（持有期内均已平仓） |

与 Phase 4.6 数字一致，确认本次修改不影响会计口径。

## 6. 仍存在的已知简化（后续）

1. `unrealized_*` 仅在有仓位卡到期末（长期停牌/连续跌停顺延）时非零；常规配置多为 0。
2. `zfill(6)` 假设 `prices`/`flags` 键为 6 位标准代码；若上游用非标准键需先归一。
3. 停牌到期仅登记 maturity，不在停牌期间评估 stop/take（停牌无成交，符合实盘）。
4. 其余简化同 [reports/backtest_engine_audit_phase46.md](backtest_engine_audit_phase46.md)（固定槽位不加仓、无 ST/行业数据、单年 2025、未与 ml_dataset 宽表直连）。
