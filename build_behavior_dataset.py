"""
Build a model-ready behavior dataset from Level-2 operation parquet files.

Inputs:
  - level2_ops_*.parquet from run_level2_archive_day.py
  - optional daily price parquet/csv with stock_code,date,open,close

Output:
  - behavior_train_samples.parquet

This script is intentionally simple. It standardizes feature columns first; price
labels can be attached when daily price data is ready.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "total_amount_wan",
    "avg_price",
    "order_count",
    "time_span_min",
    "start_time",
    "end_time",
    "buy_volume_wan",
    "price_min",
    "price_max",
    "vwap_deviation_pct",
    "avg_order_size_wan",
    "median_order_qty",
    "qty_cv",
    "mid_time_sec",
    "matched_orders",
    "order_interval_std",
    "order_hhi",
    "participation_rate",
    "price_range_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build behavior training samples.")
    parser.add_argument("--ops", nargs="+", required=True, help="Input ops parquet files or glob patterns.")
    parser.add_argument("--prices", default="", help="Optional daily prices parquet/csv.")
    parser.add_argument("--horizons", default="5,10,20", help="Forward return horizons in trading rows.")
    parser.add_argument("--output", default="data/processed/behavior_train_samples.parquet")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            files.extend(matches)
        else:
            files.append(Path(pattern))
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_ops(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_prices(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    price_path = Path(path)
    if not price_path.exists():
        raise FileNotFoundError(price_path)
    # 支持目录：加载所有 parquet 文件并合并
    if price_path.is_dir():
        frames = []
        for f in sorted(price_path.glob("*.parquet")):
            df = pd.read_parquet(f)
            stock = f.stem  # e.g. "000001"
            if "stock_code" not in df.columns:
                df["stock_code"] = stock
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        prices = pd.concat(frames, ignore_index=True)
    elif price_path.suffix.lower() == ".parquet":
        prices = pd.read_parquet(price_path)
    else:
        prices = pd.read_csv(price_path)
    required = {"stock_code", "date", "open", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price file missing columns: {sorted(missing)}")
    # 参与率和价格冲击需要用到的额外列
    extra_available = {c for c in ["volume", "high", "low"] if c in prices.columns}
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.sort_values(["stock_code", "date"])


def attach_forward_returns(samples: pd.DataFrame, prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    if samples.empty or prices.empty:
        return samples

    rows = []
    for stock, stock_samples in samples.groupby("stock_code"):
        px = prices[prices["stock_code"] == stock].sort_values("date").reset_index(drop=True)
        if px.empty:
            rows.append(stock_samples)
            continue
        date_to_idx = {d: i for i, d in enumerate(px["date"])}
        enriched = stock_samples.copy()
        for horizon in horizons:
            values = []
            wins = []
            for date in enriched["date"]:
                idx = date_to_idx.get(date)
                if idx is None or idx + horizon >= len(px):
                    values.append(pd.NA)
                    wins.append(pd.NA)
                    continue
                entry = px.loc[idx + 1, "open"] if idx + 1 < len(px) else px.loc[idx, "close"]
                future = px.loc[idx + horizon, "close"]
                ret = (future - entry) / entry if entry else pd.NA
                values.append(ret)
                wins.append(ret > 0 if pd.notna(ret) else pd.NA)
            enriched[f"ret_{horizon}d"] = values
            enriched[f"win_{horizon}d"] = wins

        # 参与率 = 集群买入量(股) / 当日成交量(股)
        if "volume" in px.columns and "buy_volume_wan" in enriched.columns:
            rates = []
            for _, row in enriched.iterrows():
                idx = date_to_idx.get(row["date"])
                if idx is not None:
                    daily_vol = px.loc[idx, "volume"]
                    cluster_vol_shares = row["buy_volume_wan"] * 10000
                    rates.append(cluster_vol_shares / daily_vol if daily_vol > 0 else pd.NA)
                else:
                    rates.append(pd.NA)
            enriched["participation_rate"] = rates

        # 价格冲击 = 集群价格区间 / 当日振幅
        if {"high", "low"}.issubset(px.columns) and {"price_min", "price_max"}.issubset(enriched.columns):
            ratios = []
            for _, row in enriched.iterrows():
                idx = date_to_idx.get(row["date"])
                if idx is not None:
                    day_range = px.loc[idx, "high"] - px.loc[idx, "low"]
                    cluster_range = row["price_max"] - row["price_min"]
                    ratios.append(cluster_range / day_range if day_range > 0 else pd.NA)
                else:
                    ratios.append(pd.NA)
            enriched["price_range_ratio"] = ratios

        rows.append(enriched)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    ops_files = expand_inputs(args.ops)
    samples = load_ops(ops_files)
    if samples.empty:
        raise SystemExit("No operation rows found.")

    for col in FEATURE_COLUMNS:
        if col not in samples.columns:
            samples[col] = pd.NA

    prices = load_prices(args.prices)
    samples = attach_forward_returns(samples, prices, horizons)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(output, index=False)
    print(f"input_files={len(ops_files)}")
    print(f"rows={len(samples)}")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
