# Level-2 Quant Workflow

## Current Resource Rule

If Qwen2.5-VL training is running on the Windows machine, do not run full-day or
full-month Level-2 jobs. Only run code checks and small samples.

Allowed while Qwen is running:

```bash
python -m pytest -q tests
python run_level2_archive_day.py --archive ... --date 20250102 --stocks 000001.SZ
python run_level2_archive_day.py --archive ... --date 20250102 --max-stocks 20
```

Do not run while Qwen is running:

```bash
python run_level2_archive_day.py --archive ... --date 20250102
python build_behavior_dataset.py --ops "data/processed/level2_ops_*.parquet" --prices ...
python train_lgbm.py --data ...
```

## Data Layout

Raw Level-2 archives on Windows:

```text
C:\Users\1\Desktop\2025\202501\20250102.7z
C:\Users\1\Desktop\2025\202501\20250103.7z
...
```

Project:

```text
C:\Users\1\Desktop\institution-alpha
```

Archive contents:

```text
20250102/000001.SZ/行情.csv
20250102/000001.SZ/逐笔委托.csv
20250102/000001.SZ/逐笔成交.csv
```

## Matching Rule

For this 2025 data, `逐笔委托.委托编号` is often zero. Matching must use:

```text
逐笔委托.交易所委托号 <-> 逐笔成交.叫买序号 / 叫卖序号
```

The reader now auto-selects the match key:

```text
use 委托编号 if usable
otherwise use 交易所委托号
```

## Phase 1: Generate Level-2 Operation Features

Small validation:

```bash
cd C:\Users\1\Desktop\institution-alpha
.venv\Scripts\python.exe run_level2_archive_day.py ^
  --archive C:\Users\1\Desktop\2025\202501\20250102.7z ^
  --date 20250102 ^
  --max-stocks 20 ^
  --output data\processed\level2_ops_20250102_first20.parquet
```

Full day, only after Qwen training is done:

```bash
cd C:\Users\1\Desktop\institution-alpha
.venv\Scripts\python.exe run_level2_archive_day.py ^
  --archive C:\Users\1\Desktop\2025\202501\20250102.7z ^
  --date 20250102 ^
  --output data\processed\level2_ops_20250102.parquet
```

## Phase 2: Build Training Samples

Requires daily price data with:

```text
stock_code,date,open,close
```

Then:

```bash
.venv\Scripts\python.exe build_behavior_dataset.py ^
  --ops "data/processed/level2_ops_202501*.parquet" ^
  --prices data\processed\daily_prices.parquet ^
  --output data\processed\behavior_train_samples.parquet
```

## Phase 3: Train LightGBM

Regression target:

```bash
.venv\Scripts\python.exe train_lgbm.py ^
  --data data\processed\behavior_train_samples.parquet ^
  --target ret_5d ^
  --task regression ^
  --output-dir models
```

Classification target:

```bash
.venv\Scripts\python.exe train_lgbm.py ^
  --data data\processed\behavior_train_samples.parquet ^
  --target win_5d ^
  --task classification ^
  --output-dir models
```
