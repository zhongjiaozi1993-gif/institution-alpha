# archive/legacy_scripts — 历史实验脚本（仅作追溯）

这些脚本是早期一次性实验 / 评估 / 排查产物，**不属于当前主流程**，不保证可运行、不维护、不接入 Phase 0-4.5 的
universe / label / tradable_flags / ml_dataset / 回测引擎。

保留原因：追溯当时的分析思路与结论。**新开发请勿依赖或 import 本目录任何文件。**

主流程脚本在 `scripts/`（`build_universe.py` / `build_labels.py` / `build_tradable_flags.py` /
`build_ml_dataset.py` / `run_alpha191_validation.py` / `run_signal_backtest.py`）。

## 清单

| 脚本 | 原用途（追溯） |
|------|------|
| `audit_backtest_consistency.py` | 一次性回测一致性审计 |
| `backtest_enhanced_risk.py` | 风险参数网格实验（含单位错误的指数过滤，已被 signal_backtester 取代） |
| `backtest_insttracker_signals.py` | 机构信号回测早期版（被 signal_backtester 取代） |
| `batch_run_v4_v6.py` / `batch_run_v4_v6_heavy.py` | Sofia v4-v6 批量跑，被验证框架取代 |
| `case_study_000547.py` | 单票个案分析 |
| `check_2026_level2_ops_quality.py` | Level-2 输出质量一次性检查 |
| `check_oos_data_availability.py` | 样本外数据可用性一次性检查 |
| `evaluate_pipeline_fusion.py` | 早期融合评估一次性脚本 |
| `extract_raw_for_v6.py` | v6 原始数据抽取，被 sofia_v6 取代 |
| `robustness_validation.py` / `run_unified_robustness_grid.py` | 稳健性网格实验，产物已落地 |
| `run_v6_pipeline_300.py` | v6 300 股跑批，被新 pipeline 取代 |

## 处置建议

确认无追溯价值后可整体删除本目录；在此之前不要把这些脚本移回 `scripts/` 主流程。
