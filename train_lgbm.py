"""
Train a LightGBM model on behavior_train_samples.parquet.

Default split is chronological by date. This is required for quant work: random
splits leak future market regimes into validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score


DEFAULT_FEATURES = [
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
    parser = argparse.ArgumentParser(description="Train LightGBM on Level-2 behavior samples.")
    parser.add_argument("--data", default="data/processed/behavior_train_samples.parquet")
    parser.add_argument("--target", default="ret_5d")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--min-rows", type=int, default=100)
    return parser.parse_args()


def chronological_split(df: pd.DataFrame, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].unique())
    split_idx = max(1, int(len(dates) * (1 - valid_ratio)))
    split_date = dates[split_idx]
    train = df[df["date"] < split_date].copy()
    valid = df[df["date"] >= split_date].copy()
    return train, valid


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def classification_metrics(y_true, y_prob) -> dict:
    y_pred = y_prob >= 0.5
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if len(set(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_parquet(data_path)
    if "date" not in df.columns:
        raise ValueError("data must contain date column")
    df["date"] = pd.to_datetime(df["date"])

    requested = [x.strip() for x in args.features.split(",") if x.strip()]
    missing = [col for col in requested + [args.target] if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    # 自动过滤：去掉数据中全空的列（如指纹列在旧版缓存中不存在）
    usable = [c for c in requested
              if c in df.columns and df[c].notna().sum() >= args.min_rows]
    dropped = [c for c in requested if c not in usable]
    if dropped:
        print(f"警告: 以下特征列有效数据不足，已跳过: {dropped}")
    if not usable:
        raise ValueError("没有可用的特征列（所有列缺失率过高）")
    features = usable

    data = df.dropna(subset=features + [args.target]).copy()
    if len(data) < args.min_rows:
        raise ValueError(f"not enough rows: {len(data)} < {args.min_rows}")

    train_df, valid_df = chronological_split(data, args.valid_ratio)
    x_train, y_train = train_df[features], train_df[args.target]
    x_valid, y_valid = valid_df[features], valid_df[args.target]

    if args.task == "classification":
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_valid)[:, 1]
        metrics = classification_metrics(y_valid, pred)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        metrics = regression_metrics(y_valid, pred)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lgbm_{args.target}.pkl"
    metrics_path = out_dir / f"lgbm_{args.target}_metrics.json"
    importance_path = out_dir / f"lgbm_{args.target}_feature_importance.csv"

    joblib.dump({"model": model, "features": features, "target": args.target, "task": args.task}, model_path)

    payload = {
        "task": args.task,
        "target": args.target,
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "valid_start": str(valid_df["date"].min().date()),
        "valid_end": str(valid_df["date"].max().date()),
        "metrics": metrics,
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(importance_path, index=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
