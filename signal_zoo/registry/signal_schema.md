# Signal Schema

> 所有 Signal 必须遵循的统一接口规范。v1.0, 2026-07-09。

---

## 一、设计原则

1. **一个 Signal = 一个可验证的 Alpha 假设。**
2. **同一接口**：不论数据源（Daily/Level2/Fundamental），外部调用方式一致。
3. **验证优先**：每个 Signal 必须通过 Validation Pipeline 才能进入 Candidate 状态。
4. **可组合**：多个 Signal 经 Model 层组合为最终策略。

---

## 二、接口定义

```python
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class Signal(ABC):
    """所有 Signal 的抽象基类。"""

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """在训练集上拟合（如需要）。纯规则型 Signal 可为空操作。"""
        ...

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """对输入数据生成信号值。返回 DataFrame，index=date, columns=stock_code。"""
        ...

    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """便捷方法：先 fit 再 transform。"""
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    @abstractmethod
    def validate(self, data: pd.DataFrame, prices: dict, **kwargs) -> dict:
        """通过 Validation Pipeline 评估信号质量。
        
        返回 dict 至少包含:
        - IC / RankIC
        - group_return
        - win_rate
        - excess_return
        - stability
        - oot_result
        """
        ...

    @abstractmethod
    def metadata(self) -> dict:
        """返回 Signal 元信息，与 signal_registry.csv 对应。
        
        Returns:
            dict with keys: signal_id, signal_name, category, source,
            data_requirement, status, validation_status, notes
        """
        ...
```

---

## 三、输出规范

### 3.1 `transform()` 输出格式

```text
              stock_001  stock_002  ...  stock_N
2025-01-02     0.023     -0.015    ...   0.008
2025-01-03     0.018      0.022    ...  -0.012
...
```

- **index**: `date` (datetime)
- **columns**: `stock_code` (6位字符串)
- **values**: 信号值（建议 Z-score 标准化后输出，方向对齐：正值 = 看多）

### 3.2 `validate()` 输出格式

```python
{
    "signal_id": "Signal004",
    "IC_mean": 0.035,
    "IC_std": 0.12,
    "ICIR": 0.29,
    "RankIC_mean": 0.042,
    "RankIC_std": 0.11,
    "RankICIR": 0.38,
    "group_return_top": 0.0212,      # 多头组平均收益
    "group_return_bottom": -0.0083,  # 空头组平均收益
    "win_rate_5d": 0.595,
    "excess_return_5d": 0.0114,
    "stability": 0.11,
    "oot_passed": True,
    "universe": "Top100",
    "universe_n_stocks": 100,
}
```

---

## 四、状态流转

```
Research ──validation pass──> Candidate ──OOS pass──> Production
    │                              │                      │
    └──validation fail──> Archived│                      │
                                   └──OOS fail──> Archived│
                                                           │
                                ────degraded──> Archived───┘
```

- **Research**: 新 Signal，未验证或验证中
- **Candidate**: 样本内验证通过，等待 OOS
- **Production**: OOS 通过，可进入组合层
- **Archived**: 验证或 OOS 失败，归档保留代码但不参与组合

---

## 五、分类体系

| Category | 数据源 | 典型 Signal | 时间粒度 |
|----------|--------|-------------|----------|
| Price | Daily OHLCV | Alpha191, Momentum, Volatility | 日频 |
| Order Flow | Level2 逐笔/快照 | OFI, Queue Imbalance, Cancel Ratio | 分钟/日聚合 |
| Microstructure | Level2 聚类/拆单 | DBSCAN BUY, VWAP Pressure | 日聚合 |
| Fundamental | 财报/估值 | PE, PB, ROE | 季/年频 |

---

## 六、命名规范

```
Signal{NNN}_{SHORT_NAME}

例:
  Signal004_DBSCAN_BUY
  Signal017_Alpha191_KUPFRAC
  Signal007_OFI_Basic
```

- `{NNN}`: 3位数字，从 registry 分配
- `{SHORT_NAME}`: PascalCase，简明描述信号内容

---

## 七、与现有代码的关系

当前流水线（`scale_oot_300.py` 等）是 Validation Pipeline 的前身。

迁移路径：
1. 现有 `scale_oot_300.py` 的 forward return / benchmark 逻辑 → `validate()` 方法
2. 现有 `v6_priority_stocks.txt` → Signal004 的 candidate_universe
3. 现有 `analyze_dbscan_stock_profile.py` 的 A/B/C/D 分组 → `validate()` 输出的分组字段
