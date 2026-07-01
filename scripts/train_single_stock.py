"""
Train a predictive model for 002516 using Level-2 features.
Labels: forward N-day returns (binary: up/down, or regression: return %)
Features: 34 enriched Level-2 microstructure features
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
STOCK = "002516"
DATA_DIR = PROJECT / "data" / "single_stock" / STOCK

# ═══════════════════════════════════════════════════════════════
# 1. Build labeled dataset
# ═══════════════════════════════════════════════════════════════

def build_labeled_dataset() -> pd.DataFrame:
    """Join Level-2 features with forward returns from price data."""
    feat_dir = DATA_DIR / "features"
    price_path = DATA_DIR / "price_daily.csv"

    if not price_path.exists():
        print("Price data not found. Download first.")
        return pd.DataFrame()

    # Load features
    feat_files = sorted(feat_dir.glob("features_*.csv"))
    if not feat_files:
        # Try the all_features summary
        all_feat = feat_dir / "all_features.csv"
        if all_feat.exists():
            features = pd.read_csv(all_feat)
        else:
            print("No feature files found")
            return pd.DataFrame()
    else:
        features = pd.concat([pd.read_csv(f) for f in feat_files], ignore_index=True)

    # Load prices
    prices = pd.read_csv(price_path)
    prices["日期"] = pd.to_datetime(prices["日期"])
    prices = prices.sort_values("日期").reset_index(drop=True)

    # Convert feature dates (stored as "20250102" string)
    features["date"] = pd.to_datetime(features["date"].astype(str), format="%Y%m%d")

    # Merge with price on date
    merged = features.merge(prices, left_on="date", right_on="日期", how="left")
    if merged.empty:
        print("No matching dates between features and prices")
        return pd.DataFrame()

    # Forward returns: signal day T, enter at T+1 open, exit at T+h close.
    horizons = [1, 3, 5, 10, 20]
    date_to_idx = {d: i for i, d in enumerate(prices["日期"])}

    for h in horizons:
        returns = []
        for signal_date in merged["date"]:
            idx = date_to_idx.get(signal_date)
            if idx is None or idx + 1 >= len(prices) or idx + h >= len(prices):
                returns.append(np.nan)
                continue
            entry = prices.loc[idx + 1, "开盘"]
            exit_price = prices.loc[idx + h, "收盘"]
            returns.append((exit_price / entry - 1) * 100 if entry else np.nan)
        merged[f"fwd_{h}d_return"] = returns
        merged[f"fwd_{h}d_up"] = (merged[f"fwd_{h}d_return"] > 0).astype("Int64")

    # Keep rows valid for all modeled horizons so train/test slices align.
    merged = merged.dropna(subset=[f"fwd_{h}d_return" for h in horizons])
    merged = merged.sort_values("date").reset_index(drop=True)

    print(f"Labeled dataset: {len(merged)} rows × {len(merged.columns)} cols")
    print(f"Target distribution (5d up): {merged['fwd_5d_up'].mean():.1%}")

    return merged


# ═══════════════════════════════════════════════════════════════
# 2. Feature selection
# ═══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # A. Dynamic big/small
    "dynamic_big_threshold_yuan", "big_order_count", "big_order_ratio",
    "big_order_volume_ratio",
    # B. Order aggressiveness
    "aggressive_order_ratio", "avg_price_aggressiveness",
    "buy_order_count", "sell_order_count", "buy_sell_order_ratio",
    # C. Execution quality
    "fill_rate", "avg_fill_ratio", "partial_fill_ratio",
    "avg_buy_slippage_bps", "slippage_std_bps",
    # D. Temporal patterns
    "morning_volume_ratio", "opening_30min_ratio", "closing_30min_ratio",
    # E. VPIN imbalance
    "trade_buy_volume_ratio", "trade_imbalance",
    "vpin_mean", "vpin_std", "vpin_max",
    # F. Market impact
    "intraday_range_pct",
    # G. Long/short
    "median_trades_per_order", "long_order_ratio",
    # H. Cluster features
    "n_clusters", "total_cluster_amount_wan", "total_cluster_orders",
    "avg_cluster_span_min", "max_cluster_amount_wan",
    "clusters_per_hour", "buy_cluster_ratio",
]


def prepare_xy(df: pd.DataFrame, target_horizon: int = 10):
    """Split into X (features) and y (T+1 entry forward return)."""
    target_col = f"fwd_{target_horizon}d_return"
    clean = df.dropna(subset=[target_col]).copy()
    available = [c for c in FEATURE_COLS if c in clean.columns]
    X = clean[available].copy()
    y = clean[target_col].values

    # Fill NaN features with training-safe column medians.
    X = X.fillna(X.median(numeric_only=True))

    return X, y, available


# ═══════════════════════════════════════════════════════════════
# 3. Model training & evaluation
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train LightGBM (or fallback to linear) and evaluate out-of-sample."""
    if len(df) < 20:
        print(f"Only {len(df)} samples — need more data for ML. Showing correlations instead.")
        return simple_correlation_analysis(df)

    X, y, feature_names = prepare_xy(df, target_horizon=10)

    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    # Time-series cross-validation: expanding window
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error

    results = {"feature_names": feature_names, "horizon": 10}

    # Simple train/test split (last 20% as test)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = df["date"].iloc[split:]

    print(f"Train: {len(X_train)} days, Test: {len(X_test)} days")

    if has_lgb:
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        results["importance"] = importance
    else:
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Coefficients as "importance"
        importance = pd.DataFrame({
            "feature": feature_names,
            "coef": model.coef_,
        }).sort_values("coef", key=abs, ascending=False)
        results["importance"] = importance

    # Metrics
    results["r2"] = r2_score(y_test, y_pred)
    results["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
    results["corr"] = np.corrcoef(y_test, y_pred)[0, 1]

    # Directional accuracy
    dir_acc = np.mean((y_test > 0) == (y_pred > 0))
    results["directional_accuracy"] = dir_acc

    # Simple strategy: buy when prediction > threshold
    for threshold_pct in [30, 50, 70]:
        percentile = np.percentile(y_pred, threshold_pct)
        buy_mask = y_pred >= percentile
        if buy_mask.sum() > 0:
            strat_return = y_test[buy_mask].mean()
            results[f"strat_top{100-threshold_pct}pct_return"] = strat_return

    # Always-buy baseline
    results["baseline_return"] = y_test.mean()

    return results


def simple_correlation_analysis(df: pd.DataFrame) -> dict:
    """When we have too few samples, just compute correlations."""
    X, y, feature_names = prepare_xy(df, target_horizon=10)

    correlations = {}
    for col in feature_names:
        if col in X.columns and X[col].std() > 0:
            corr = np.corrcoef(X[col].values, y)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nFeature correlations with 10-day forward return:")
    for name, corr in sorted_corr[:15]:
        bar = "█" * int(abs(corr) * 50)
        sign = "+" if corr > 0 else "-"
        print(f"  {sign}{bar} {name}: {corr:.3f}")

    return {"correlations": dict(sorted_corr), "horizon": 10}


# ═══════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════

def main():
    df = build_labeled_dataset()
    if df.empty:
        print("No data available. Run single_stock_pipeline.py first.")
        return

    results = train_and_evaluate(df)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if "r2" in results:
        print(f"R²: {results['r2']:.4f}")
        print(f"RMSE: {results['rmse']:.2f}%")
        print(f"Correlation: {results['corr']:.4f}")
        print(f"Directional accuracy: {results['directional_accuracy']:.1%}")
        print(f"Baseline (always long): {results['baseline_return']:.2f}%")
        for k, v in results.items():
            if k.startswith("strat_"):
                print(f"  {k}: {v:.2f}%")

    if "importance" in results:
        imp = results["importance"]
        print(f"\nTop 10 features:")
        for _, row in imp.head(10).iterrows():
            val_col = "importance" if "importance" in imp.columns else "coef"
            print(f"  {row['feature']}: {row[val_col]:.4f}")

    if "correlations" in results:
        pass  # Already printed

    # ═════════════════════════════════════════════════════════
    # Multi-horizon comparison
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MULTI-HORIZON TEST")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'Dir Acc':>8} {'Top30%':>8} {'Baseline':>10} {'Edge':>8}")
    print(f"{'-'*50}")

    for h in [1, 3, 5, 10, 20]:
        X, y, _ = prepare_xy(df, target_horizon=h)
        if len(y) == 0:
            continue
        split = int(len(df) * 0.8)
        y_test = y[split:]

        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_child_samples=5, subsample=0.8, random_state=42, verbose=-1,
            )
            model.fit(X.iloc[:split], y[:split])
            y_pred = model.predict(X.iloc[split:])
        except ImportError:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X.iloc[:split], y[:split])
            y_pred = model.predict(X.iloc[split:])

        dir_acc = np.mean((y_test > 0) == (y_pred > 0))
        p70 = np.percentile(y_pred, 70)
        top30_ret = y_test[y_pred >= p70].mean() if (y_pred >= p70).sum() > 0 else 0
        baseline = y_test.mean()
        edge = top30_ret - baseline
        print(f"{h}d        {dir_acc:>7.1%}  {top30_ret:>7.2f}%  {baseline:>9.2f}%  {edge:>7.2f}%")

    # ═════════════════════════════════════════════════════════
    # Cumulative return curve (10-day horizon)
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CUMULATIVE RETURNS (10-day horizon, test set)")
    print(f"{'='*60}")

    X, y, _ = prepare_xy(df, target_horizon=10)
    split = int(len(df) * 0.8)
    y_test = y[split:]
    dates_test = df["date"].iloc[split:]

    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, random_state=42, verbose=-1,
        )
        model.fit(X.iloc[:split], y[:split])
        y_pred = model.predict(X.iloc[split:])
    except ImportError:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X.iloc[:split], y[:split])
        y_pred = model.predict(X.iloc[split:])

    # Three strategies
    p70 = np.percentile(y_pred, 70)
    p50 = np.percentile(y_pred, 50)

    top30_mask = y_pred >= p70
    bottom30_mask = y_pred <= np.percentile(y_pred, 30)
    always_long = np.ones(len(y_test), dtype=bool)

    cum_ret = 0
    cum_top30 = 0
    cum_bottom30 = 0
    cum_always = 0

    print(f"\n{'Date':<12} {'Pred':>7} {'Actual':>7} {'Top30':>7} {'Bot30':>7} {'Always':>7}")
    print(f"{'-'*60}")

    for i in range(len(y_test)):
        cum_always += y_test[i]
        if top30_mask[i]:
            cum_top30 += y_test[i]
        if bottom30_mask[i]:
            cum_bottom30 += y_test[i]

        if i % 7 == 0 or i == len(y_test) - 1:  # Show weekly
            d = pd.Timestamp(dates_test.iloc[i]).strftime("%m-%d") if hasattr(dates_test.iloc[i], 'strftime') else str(dates_test.iloc[i])[:10]
            print(f"{d:<12} {y_pred[i]:>6.2f}% {y_test[i]:>6.2f}% "
                  f"{cum_top30:>6.2f}% {cum_bottom30:>6.2f}% {cum_always:>6.2f}%")

    n_days = len(y_test)
    n_top = top30_mask.sum()
    n_bottom = bottom30_mask.sum()

    print(f"\nFinal ({n_days} test days):")
    print(f"  Always long: {cum_always:.2f}% (invested {n_days} days)")
    print(f"  Top 30% pred: {cum_top30:.2f}% (invested {n_top} days, "
          f"{(cum_top30/n_top*250) if n_top else 0:.1f}% ann)")
    print(f"  Bottom 30% pred: {cum_bottom30:.2f}% (invested {n_bottom} days)")
    print(f"  Long-Short spread: {cum_top30 - cum_bottom30:.2f}%")


if __name__ == "__main__":
    main()
