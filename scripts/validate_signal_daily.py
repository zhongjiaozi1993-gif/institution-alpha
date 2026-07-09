"""Sprint 2: Minimal single-signal validation.

Computes IC, RankIC, quintile returns, win rate, coverage for one Signal.
Usage:
    python scripts/validate_signal_daily.py \\
        --signal-file data/processed/signals/price_alpha191/signal017.parquet \\
        --output-prefix signal017
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT / "data" / "daily"
SIGNAL_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191"
VAL_DIR = PROJECT / "data" / "processed" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

FWD_HORIZONS = [1, 3, 5, 10, 20]
N_QUINTILES = 5


def load_prices(stock_codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """Load daily close prices for a list of stocks."""
    prices = {}
    for code in stock_codes:
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df.sort_values("date").reset_index(drop=True)
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        prices[code] = df[mask].copy()
    return prices


def attach_forward_returns(signal_df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Attach forward returns at multiple horizons. No look-ahead bias."""
    df = signal_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    for h in FWD_HORIZONS:
        df[f"fwd_{h}d"] = np.nan
        df[f"win_{h}d"] = np.nan

    for code in df["stock_code"].unique():
        if code not in prices:
            continue
        pdf = prices[code]
        close_series = pdf.set_index("date")["close"].sort_index()
        dates_arr = close_series.index

        mask = df["stock_code"] == code
        idxs = df.loc[mask].index
        for i in idxs:
            t = df.loc[i, "trade_date"]
            if t not in dates_arr:
                continue
            pos = dates_arr.get_loc(t)
            for h in FWD_HORIZONS:
                tgt_pos = pos + h
                if tgt_pos < len(dates_arr):
                    c_t = close_series.iloc[pos]
                    c_fwd = close_series.iloc[tgt_pos]
                    fwd = (c_fwd / c_t - 1) * 100
                    df.loc[i, f"fwd_{h}d"] = fwd
                    df.loc[i, f"win_{h}d"] = 1.0 if fwd > 0 else 0.0

    return df


def compute_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily IC and RankIC (Spearman) per horizon."""
    rows = []
    for date, g in df.groupby("trade_date"):
        if len(g) < 5:
            continue
        for h in FWD_HORIZONS:
            col = f"fwd_{h}d"
            valid = g[["signal_value", col]].dropna()
            if len(valid) < 5:
                continue
            pearson = valid["signal_value"].corr(valid[col], method="pearson")
            spearman = valid["signal_value"].corr(valid[col], method="spearman")
            rows.append({
                "trade_date": date,
                "horizon": h,
                "IC": pearson,
                "RankIC": spearman,
                "n_stocks": len(valid),
            })
    return pd.DataFrame(rows)


def compute_quintile_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward return by signal quintile per date."""
    rows = []
    for date, g in df.groupby("trade_date"):
        g_clean = g.dropna(subset=["signal_value"])
        if len(g_clean) < N_QUINTILES * 2:
            continue
        g_clean["quintile"] = pd.qcut(g_clean["signal_value"], N_QUINTILES, labels=False, duplicates="drop")
        for h in FWD_HORIZONS:
            col = f"fwd_{h}d"
            q_means = g_clean.groupby("quintile")[col].mean()
            for q, v in q_means.items():
                rows.append({
                    "trade_date": date,
                    "horizon": h,
                    "quintile": int(q),
                    "avg_fwd": v,
                    "n_stocks": (g_clean["quintile"] == q).sum(),
                })
    return pd.DataFrame(rows)


def validate_signal(signal_file: str, output_prefix: str,
                    start_date: str = "2025-01-01", end_date: str = "2025-12-31"):
    """Run full validation for one Signal."""
    sig = pd.read_parquet(signal_file)
    sig["trade_date"] = pd.to_datetime(sig["trade_date"])
    sig = sig[(sig["trade_date"] >= start_date) & (sig["trade_date"] <= end_date)]

    signal_id = sig["signal_id"].iloc[0] if "signal_id" in sig.columns else "Unknown"
    signal_name = sig["signal_name"].iloc[0] if "signal_name" in sig.columns else "Unknown"

    stock_codes = sorted(sig["stock_code"].unique())
    n_stocks = len(stock_codes)
    n_dates = sig["trade_date"].nunique()
    total_rows = len(sig)
    missing_pct = sig["signal_value"].isna().mean() * 100

    print(f"  Signal: {signal_id} ({signal_name})")
    print(f"  Universe: {n_stocks} stocks, {n_dates} dates, {total_rows} rows")
    print(f"  Signal missing: {missing_pct:.1f}%")

    # Load prices and attach fwd returns
    prices = load_prices(stock_codes, start_date, end_date)
    df = attach_forward_returns(sig, prices)

    # Coverage: % of signal rows that have valid fwd return
    coverage = {}
    for h in FWD_HORIZONS:
        col = f"fwd_{h}d"
        coverage[h] = df[col].notna().mean() * 100

    # IC analysis
    ic_df = compute_ic(df)
    ic_summary = {}
    for h in FWD_HORIZONS:
        sub = ic_df[ic_df["horizon"] == h]
        ic_summary[h] = {
            "IC_mean": sub["IC"].mean(),
            "IC_std": sub["IC"].std(),
            "ICIR": sub["IC"].mean() / sub["IC"].std() if sub["IC"].std() > 0 else 0,
            "RankIC_mean": sub["RankIC"].mean(),
            "RankIC_std": sub["RankIC"].std(),
            "RankICIR": sub["RankIC"].mean() / sub["RankIC"].std() if sub["RankIC"].std() > 0 else 0,
            "IC_pos_pct": (sub["IC"] > 0).mean() * 100,
            "RankIC_pos_pct": (sub["RankIC"] > 0).mean() * 100,
        }

    # Quintile analysis
    q_df = compute_quintile_returns(df)
    quintile_summary = {}
    for h in FWD_HORIZONS:
        sub = q_df[q_df["horizon"] == h]
        q_means = sub.groupby("quintile")["avg_fwd"].mean()
        if 0 in q_means.index and (N_QUINTILES - 1) in q_means.index:
            top = q_means[N_QUINTILES - 1]
            bottom = q_means[0]
            spread = top - bottom
        else:
            top = bottom = spread = np.nan
        quintile_summary[h] = {
            "top_quintile_avg": top,
            "bottom_quintile_avg": bottom,
            "spread": spread,
        }

    # Win rate
    win_rate = {}
    for h in FWD_HORIZONS:
        col = f"win_{h}d"
        win_rate[h] = df[col].mean() * 100

    # Direction check: top quintile signal -> positive return?
    direction_check = {}
    for h in FWD_HORIZONS:
        sub = q_df[q_df["horizon"] == h]
        if len(sub) > 0:
            top_dates = sub[sub["quintile"] == (N_QUINTILES - 1)]["avg_fwd"]
            bottom_dates = sub[sub["quintile"] == 0]["avg_fwd"]
            direction_check[h] = {
                "top_pos_pct": (top_dates > 0).mean() * 100,
                "bottom_pos_pct": (bottom_dates > 0).mean() * 100,
            }

    # ---- Generate report ----
    rpt_path = VAL_DIR / f"{output_prefix}_validation_report.md"
    sum_path = VAL_DIR / f"{output_prefix}_validation_summary.csv"

    with open(rpt_path, "w") as f:
        f.write(f"# {signal_id}: {signal_name} — Validation Report\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("---\n\n")
        f.write("## 基本信息\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| Signal ID | {signal_id} |\n")
        f.write(f"| Signal Name | {signal_name} |\n")
        f.write(f"| Universe | {n_stocks} stocks (V0 candidates) |\n")
        f.write(f"| Period | {start_date} ~ {end_date} |\n")
        f.write(f"| Trading days | {n_dates} |\n")
        f.write(f"| Signal missing rate | {missing_pct:.1f}% |\n\n")

        f.write("## Coverage\n\n")
        f.write("| Horizon | Coverage |\n")
        f.write("|---------|----------|\n")
        for h in FWD_HORIZONS:
            f.write(f"| {h}d | {coverage[h]:.1f}% |\n")
        f.write("\n")

        f.write("## IC / RankIC\n\n")
        f.write("| Horizon | IC Mean | ICIR | RankIC Mean | RankICIR | IC>0% | RankIC>0% |\n")
        f.write("|---------|---------|------|-------------|----------|-------|------------|\n")
        for h in FWD_HORIZONS:
            s = ic_summary[h]
            f.write(f"| {h}d | {s['IC_mean']:.4f} | {s['ICIR']:.2f} | "
                    f"{s['RankIC_mean']:.4f} | {s['RankICIR']:.2f} | "
                    f"{s['IC_pos_pct']:.0f}% | {s['RankIC_pos_pct']:.0f}% |\n")
        f.write("\n")

        f.write("## Quintile Returns\n\n")
        f.write("| Horizon | Top Q | Bottom Q | Spread |\n")
        f.write("|---------|-------|----------|--------|\n")
        for h in FWD_HORIZONS:
            s = quintile_summary[h]
            f.write(f"| {h}d | {s['top_quintile_avg']:+.3f}% | "
                    f"{s['bottom_quintile_avg']:+.3f}% | {s['spread']:+.3f}% |\n")
        f.write("\n")

        f.write("## Win Rate\n\n")
        f.write("| Horizon | Win Rate |\n")
        f.write("|---------|----------|\n")
        for h in FWD_HORIZONS:
            f.write(f"| {h}d | {win_rate[h]:.1f}% |\n")
        f.write("\n")

        f.write("## Direction Check\n\n")
        f.write("| Horizon | TopQ >0% | BottomQ >0% |\n")
        f.write("|---------|----------|-------------|\n")
        for h in FWD_HORIZONS:
            if h in direction_check:
                d = direction_check[h]
                f.write(f"| {h}d | {d['top_pos_pct']:.0f}% | {d['bottom_pos_pct']:.0f}% |\n")
        f.write("\n")

        # Judgment
        ic5 = ic_summary[5]
        q5 = quintile_summary[5]
        f.write("## 判断\n\n")
        checks = []
        if ic5["RankIC_mean"] > 0:
            checks.append(f"RankIC_5d = {ic5['RankIC_mean']:.4f} > 0 → 信号有正向选股能力")
        else:
            checks.append(f"RankIC_5d = {ic5['RankIC_mean']:.4f} ≤ 0 → 信号无正向选股能力")
        if not np.isnan(q5["spread"]) and q5["spread"] > 0:
            checks.append(f"Top-Bottom Spread = {q5['spread']:+.3f}% > 0 → 多空分组有效")
        else:
            checks.append(f"Top-Bottom Spread = {q5['spread']:+.3f}% ≤ 0 → 分组无效或反向")
        if win_rate[5] > 52:
            checks.append(f"Win Rate_5d = {win_rate[5]:.1f}% > 52% → 达到候选标准")
        else:
            checks.append(f"Win Rate_5d = {win_rate[5]:.1f}% ≤ 52% → 未达候选标准")
        for c in checks:
            f.write(f"- {c}\n")
        f.write("\n")

        f.write("## 结论\n\n")
        if ic5["RankICIR"] > 0.3 and win_rate[5] > 52:
            f.write("**推荐进入 Candidate。**\n")
        elif ic5["RankIC_mean"] > 0:
            f.write("**有研究价值但暂不满足候选标准。** 需观察更长时间或更大股票池。\n")
        else:
            f.write("**暂不推荐。** 信号在当前股票池上无明显预测能力。\n")

    print(f"  Report: {rpt_path}")

    # Summary CSV for comparison across signals
    summary = {
        "signal_id": signal_id,
        "signal_name": signal_name,
        "n_stocks": n_stocks,
        "n_dates": n_dates,
        "missing_pct": missing_pct,
    }
    for h in FWD_HORIZONS:
        s = ic_summary[h]
        summary[f"IC_mean_{h}d"] = s["IC_mean"]
        summary[f"RankIC_mean_{h}d"] = s["RankIC_mean"]
        summary[f"RankICIR_{h}d"] = s["RankICIR"]
        summary[f"IC_pos_pct_{h}d"] = s["IC_pos_pct"]
        qs = quintile_summary[h]
        summary[f"spread_{h}d"] = qs["spread"]
        summary[f"win_rate_{h}d"] = win_rate[h]

    pd.DataFrame([summary]).to_csv(sum_path, index=False)
    print(f"  Summary: {sum_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate a single Signal")
    parser.add_argument("--signal-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()
    validate_signal(args.signal_file, args.output_prefix, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
