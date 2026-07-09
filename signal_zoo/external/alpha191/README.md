# Alpha191 Price Signal Adapter

GTJA Alpha191 短周期量价因子适配层。公式来源 aurumq-rl (yupoet/aurumq-rl, MIT License)，pandas 重实现。

## 已接入因子

| Factor Key | Signal ID | 类别 | 公式摘要 |
|-----------|-----------|------|---------|
| gtja_002 | Signal017 | Reversal | `-Δ(((C-L)-(H-C))/(H-L), 1)` |
| gtja_004 | Signal018 | Volume-Price | MA8±STD8 vs MA2 趋势判定 + 量比门控 |
| gtja_070 | Signal019 | Volatility | `STD(amount, 6)` |
| gtja_085 | Signal020 | Momentum | `TSRANK(V/MA(V,20),20) × TSRANK(-Δ(C,7),8)` |

## 用法

```python
from signal_zoo.external.alpha191.selected_alpha191_signals import generate_all_signals, load_candidate_stocks

stocks = load_candidate_stocks()
results = generate_all_signals(stocks)
# 产出: data/processed/signals/price_alpha191/signal017~020.parquet
```

单个因子：
```python
from signal_zoo.external.alpha191.adapter import compute_signal_batch

df = compute_signal_batch(["000001", "000002"], "gtja_085")
```

## 验证

```bash
python3 scripts/validate_signal_daily.py --signal-file data/processed/signals/price_alpha191/signal020.parquet --output-prefix signal020
```

## Sprint 2 结论

13-stock Candidate V0 池 (2025) 上的表现：Signal020 (Momentum) 最优，RankIC 全正；Signal018 有长周期预测趋势；Signal019 可反用（低波动溢价）；Signal017 无效。

详见 `data/processed/validation/alpha191_price_signals_sprint2_report.md`
