# institution-alpha — 机构Alpha追踪量化系统

## 项目定位

基于龙虎榜席位级数据归因机构选股能力，生成跟买信号；中长期接入Level-2逐笔拆单聚类实现盘中机构识别。

## 技术栈

- Python 3.9+, pandas, numpy, scipy, scikit-learn
- akshare（龙虎榜+日线数据）
- pyyaml（配置）, loguru（日志）, pydantic（数据校验）
- 依赖管理：pip + pyproject.toml

## 架构分层（6层）

```
Layer 1: 数据采集    src/data/      龙虎榜(席位级)/日线(Sina)/Level-2(GB18030)
Layer 2: 机构识别    src/cluster/   拆单聚类 + 龙虎榜锚定（阶段B）
Layer 3: Alpha归因   src/alpha/     机构收益计算/画像/动态评分
Layer 4: 信号生成    src/signal/    跟买信号/共振合成/衰减
Layer 5: 交易执行    src/backtest/  回测引擎/绩效指标
Layer 6: 风控        src/risk/      市场环境/Alpha衰减/拥挤度
```

## 开发规范

- **简洁优先**：不加未请求的功能，不为一性次代码建抽象
- **数据缓存**：akshare下载的数据本地Parquet缓存，避免重复请求
- **配置集中**：所有参数在 config/settings.yaml，不在代码中硬编码
- **两阶段推进**：MVP先跑纯龙虎榜闭环，Level-2聚类为阶段B
- **编码处理**：Level-2 Wind数据为GB18030编码
- **B方案优先**：Level-2阶段先验证“行为模式 → 后续收益”，不急于识别具体机构身份

## 数据源（2026-06更新）

| 数据源 | 接口 | 状态 | 说明 |
|--------|------|------|------|
| 席位级龙虎榜 | `stock_lhb_stock_detail_em()` | 可用 | 东方财富，含每个营业部买卖金额 |
| 上榜股票列表 | `stock_lhb_detail_daily_sina()` | 可用 | 新浪，支持任意历史日期 |
| 日线后复权 | `stock_zh_a_daily()` | 可用 | Sina来源，hfq价格单位分(需/100) |
| 指数日线 | `stock_zh_index_daily()` | 可用 | Sina来源，sh000852=中证1000 |
| 营业部排名 | `stock_lhb_yytj_sina()` | 可用 | 新浪，累积统计 |
| Level-2逐笔 | 本地文件（Wind格式GB18030） | 需接入 | 用户提供，参考 l2_read.py |

**不可用API**（东方财富来源被封）：
- `stock_zh_a_hist()` — 用 `stock_zh_a_daily()` 替代
- `stock_lhb_detail_em()` 系列 — 部分可用(限近期)

**价格列名规范**（全项目统一英文）：
- 股票日线: `date, open, high, low, close, volume, amount`
- 指数日线: `date, open, high, low, close, volume`

**Level-2 列名规范**（中文，匹配原始数据）：
- 逐笔委托: `万得代码, 交易所代码, 自然日, 时间, 委托编号, 交易所委托号, 委托类型, 委托代码, 委托价格, 委托数量`
- 逐笔成交: `万得代码, 交易所代码, 自然日, 时间, 成交编号, 成交代码, 委托代码, BS标志, 成交价格, 成交数量, 叫卖序号, 叫买序号`
- 深圳成交代码=C=撤单(需过滤), 上海委托类型=D=撤单(需过滤)
- 委托成交匹配键：成交表的 `叫买序号/叫卖序号` 对应委托表的 `委托编号`，不要用 `交易所委托号`（样本里可能为0）
- 价格单位：Level-2价格字段为 `元 × 10000`
- 金额单位：原始成交金额=`价格字段 × 股数`; 转万元用 `/1e8`，转亿元用 `/1e12`
- 大单股数阈值：10万股=`10 * 10000`; 超大单股数阈值：50万股=`50 * 10000`

## 目录约定

- `data/lhb/` — 龙虎榜缓存（席位明细按日期分Parquet）
- `data/daily/` — 日线Parquet缓存，按股票代码分文件
- `data/level2/` — Level-2数据软链接
- `data/processed/` — Alpha画像、回测结果
- `notebooks/` — 探索性分析
- `tests/` — 核心模块测试

## 已知限制

- **免费LHB席位明细仅覆盖近期**：`stock_lhb_stock_detail_em` 对较早日期返回空。需每日运行积累数据。
- **Sina LHB仅1周回溯**：翻页最多8页/304条。席位明细需逐个股票查询。
- **价格数据无未来**：当日查询价格只能到当日。计算N日收益需要N天前的LHB数据。

## 002516 机构行为证据链（2026-06-30新增）

当前先做单票 `002516`，不扩全市场。核心入口：

```bash
python3 scripts/build_evidence_chain.py --stock 002516
```

离线只用本地 Level-2 行为数据：

```bash
python3 scripts/build_evidence_chain.py --stock 002516 --offline
```

核心模块在 `src/evidence/`，输出在 `data/single_stock/002516/evidence/`：

- `notable_events.csv`：重点行为日，优先看。
- `daily_evidence.csv`：全量日级机构行为证据。
- `public_evidence.csv`：龙虎榜/公告等公开源归一化结果。
- `holder_changes.csv`：基金/主要股东/流通股东的报表期持仓变化，用于滞后确认。
- `source_status.csv`：公开源抓取状态。
- `evidence_report.md`：人读摘要。

当前已确认 2025-08-13、2025-09-08、2025-09-10 为超级买入扫货/集中建仓，2025-09-11 为超级卖出/集中出货；公开龙虎榜/公告暂未匹配到 2025 这些行为日。注意：`price_daily.csv` 与 Level-2 价格疑似不同口径，收益列仅供事件参考，正式回测前需统一价格口径。

已扩展深挖公开源：东方财富新闻、Sina基金持仓、Sina主要股东、Sina流通股东、同花顺股东变动。重点滞后线索包括：2025-09-30 香港中央结算有限公司较2025-06-30减少约580.8万股，2025-12-31又较2025-09-30增加约762.2万股；野村东方国际日出东方1号资管计划在2025-09-30持有1200万股且较2025-06-30不变。后续需要区分控股权转让、指数基金调仓、北向资金变化和主动机构资金。
