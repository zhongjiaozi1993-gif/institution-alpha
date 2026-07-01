# 机构行为证据链系统

目标：先围绕单票 `002516` 把 Level-2 机构行为、市场反应、公开证据源串成一条可复核的投机证据链。

## 运行

```bash
python3 scripts/build_evidence_chain.py --stock 002516
```

离线只用本地数据：

```bash
python3 scripts/build_evidence_chain.py --stock 002516 --offline
```

## 输出

默认输出到：

```text
data/single_stock/002516/evidence/
```

文件：

- `daily_evidence.csv`：所有交易日行为证据。
- `notable_events.csv`：重点行为日，优先看这个。
- `public_evidence.csv`：龙虎榜/公告等公开源归一化结果。
- `holder_changes.csv`：基金/主要股东/流通股东的报表期持仓变化。
- `source_status.csv`：公开源抓取状态，防止误判。
- `evidence_report.md`：给人读的摘要报告。

## 字段含义

- `buy_wan` / `sell_wan` / `net_wan`：Level-2 行为簇聚合后的买卖金额，单位万元。
- `behavior_type`：行为标签，例如超级买入扫货、集中出货、多簇买入推进。
- `behavior_confidence`：规则置信度，不是机器学习概率。
- `max_op_*`：当天最大行为簇证据，包含金额、方向、时间、笔数。
- `public_event_count` / `public_events`：行为日前后公开事件匹配。
- `fwd_*_t1open_pct`：按 T+1 开盘进入的事件后收益参考。
- `share_delta`：相对上一报告期的持股数量变化，用于滞后确认。

## 当前重要限制

当前 `price_daily.csv` 与 Level-2 成交价疑似不同口径：日线价格在 40-50 区间，Level-2 价格在 5-6 区间。行为金额和方向仍可用，但收益列只适合辅助观察，正式回测前必须统一价格口径。

公开龙虎榜能验证具体席位时，会进入 `public_evidence.csv`；如果没有龙虎榜，不代表没有机构行为，只表示公开席位不可直接确认。

## CC 后续接手建议

1. 优先阅读 `notable_events.csv` 和 `evidence_report.md`。
2. 修复价格口径后，再把 `fwd_*_t1open_pct` 接入信号验证。
3. 新增大宗交易、股东户数、基金持仓、互动易/新闻题材抓取，作为新的公开证据源。
4. 把行为标签从规则升级为半监督画像：先用 002516 的 2025-08-13、2025-09-08、2025-09-10、2025-09-11 做人工锚点。
5. 对 `holder_changes.csv` 做事件解释：区分控股权转让、指数基金调仓、北向资金变化和主动资金行为。
