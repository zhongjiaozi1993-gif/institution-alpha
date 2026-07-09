# Alpha191 调研报告

> 目标: 找到可直接使用的 Alpha191（国泰君安191短周期因子）实现，避免自己重写。
> 原则: Never Reinvent the Wheel。
> 日期: 2026-07-09

---

## 一、背景

**Alpha191** 源于国泰君安 2017 年 6 月研究报告《基于短周期量价特征的多因子选股体系》，包含 191 个基于日频量价数据的 Alpha 因子。

注意区分三个常被混淆的概念：
- **Alpha101** (WorldQuant, 2015): 101 个公式化 Alpha, Zura Kakushadze 论文
- **Alpha191** (GTJA, 2017): 国泰君安 191 个短周期量价因子
- **Alpha158** (Microsoft Qlib): 158 个技术指标因子, Qlib 内置

---

## 二、可直接使用的实现

### 2.1 yupoet/aurumq-rl (推荐)

| 维度 | 评价 |
|------|------|
| 地址 | `github.com/yupoet/aurumq-rl` |
| 包含因子 | **105 Alpha101 + 191 GTJA Alpha191 = 296 因子** |
| 技术栈 | **Polars-native** 因子引擎（比 pandas 快） |
| 维护状态 | 活跃（2025-2026 持续更新） |
| License | **MIT** — 可自由使用，包括商业用途 |
| A股支持 | **是** — 专为 A 股设计，包含涨跌停处理、北向资金等 |
| 接入难度 | 中等 — 需要理解 Polars 表达式引擎，但代码结构清晰 |
| 额外优势 | GPU 训练 + ONNX CPU 推理，11 个 A 股专属因子族 |

### 2.2 meisamgh/Alpha-101-GTJA-191

| 维度 | 评价 |
|------|------|
| 地址 | `github.com/meisamgh/Alpha-101-GTJA-191` |
| 包含因子 | Alpha101 + GTJA Alpha191 |
| 技术栈 | **纯 NumPy/Pandas** |
| 维护状态 | 低（最近更新较早） |
| License | 未明确声明（风险） |
| A股支持 | 因子公式通用，但未专门适配 |
| 接入难度 | **低** — 纯 Python 脚本，可直接提取公式逻辑 |

### 2.3 SelenaMa9812/Guotai-Junan-191-Alpha

| 维度 | 评价 |
|------|------|
| 地址 | `github.com/SelenaMa9812/Guotai-Junan-191-Alpha` |
| 包含因子 | GTJA 191（专注） |
| 技术栈 | Pandas + 多因子选股框架 |
| 维护状态 | 低 |
| License | 未明确声明（风险） |
| A股支持 | **是** — 基于 A 股设计 |
| 接入难度 | 中等 — 含完整策略框架，需剥离核心因子逻辑 |

### 2.4 Microsoft Qlib Alpha158

| 维度 | 评价 |
|------|------|
| 地址 | `github.com/microsoft/qlib` |
| 包含因子 | Alpha158（158 个技术指标） + Alpha360（360 个价格序列） |
| 技术栈 | Qlib 表达式引擎 |
| 维护状态 | **活跃**（Microsoft 官方维护） |
| License | **MIT** |
| A股支持 | **是** — Qlib 原生支持 A 股数据源 |
| 接入难度 | **低** — `from qlib.contrib.data.handler import Alpha158` 即可 |
| 注意 | Alpha158 ≠ Alpha191。因子集合不同，但覆盖相似维度 |

---

## 三、推荐方案

### 首选: aurumq-rl

**理由:**
1. **MIT License** — 无法律风险
2. **296 因子**（Alpha101 + Alpha191）— 一站式覆盖
3. **Polars-native** — 性能远超 pandas 实现
4. **A 股原生** — 含涨跌停、北向资金等 A 股特有处理
5. **仍在活跃维护** — 2025-2026 持续更新

**接入方案:**

```python
# 方案 A: 直接引入因子模块（推荐）
# 从 aurumq-rl 提取 factor/ 目录，适配当前项目的数据格式

# 方案 B: 包装为统一 Signal 接口
# 每个 Alpha191 因子实现为一个 Signal 子类
class Signal017_Alpha191_KUPFRAC(Signal):
    def transform(self, data):
        # 调用 aurumq-rl 的因子计算，输出标准化信号
        ...
```

### 次选: Qlib Alpha158

如果只需要快速接入一批质量有保证的因子，Qlib 的 Alpha158 是最低摩擦的方案。158 个因子覆盖动量/波动率/量价关系等核心维度，且自带 `ZScoreNorm` 和 `Fillna` 处理。

### 不推荐: 纯脚本实现

`meisamgh/Alpha-101-GTJA-191` 和 `SelenaMa9812/Guotai-Junan-191-Alpha` 可用但 License 不明确，且维护停滞。可作为公式参考，不建议直接集成。

---

## 四、与当前框架的集成路径

```
当前状态:
  scale_oot_300.py (V6 专用验证)
  data/daily/{stock}.parquet (日线缓存)
  data/level2/ (L2 数据)

目标状态:
  signal_zoo/signals/alpha191/    ← 因子实现（来自 aurumq-rl）
  signal_zoo/validation/          ← 统一验证（validation_pipeline.md）
  signal_zoo/registry/            ← 注册 20 个 Alpha191 Signal

Step 1: clone aurumq-rl, 提取 factor/ 核心模块
Step 2: 适配数据格式 (aurumq-rl 的 OHLCV → 我们的 daily parquet)
Step 3: 为每个因子实现 Signal 接口
Step 4: 跑 Validation Pipeline
Step 5: 通过验证的因子 → 注册为 Candidate
```

---

## 五、结论

| 问题 | 答案 |
|------|------|
| 可直接使用的实现？ | aurumq-rl（296因子，MIT）+ Qlib Alpha158 |
| 哪个维护最好？ | Qlib（Microsoft官方）> aurumq-rl（活跃社区） |
| License 是否允许？ | MIT（aurumq-rl, Qlib）— 均允许使用 |
| 是否支持 A 股？ | aurumq-rl 原生支持，Qlib 通过 akshare 适配 |
| 是否容易接入？ | Qlib 最易（pip install pyqlib），aurumq-rl 需适配数据格式 |
| 推荐方案 | **aurumq-rl 做主源 + Qlib Alpha158 做补充** |
