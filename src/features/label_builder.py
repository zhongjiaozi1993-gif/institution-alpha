"""统一 label 生成器（无未来函数）。

label 定义（用户规范）:
    label_hd = T+1 open → T+(1+h) open 的收益
    即 open[T+1+h] / open[T+1] - 1

关键点:
- 基点为信号日 T；信号 T 日盘后可得，真实入场 T+1 开盘，故用 T+1 open 起算，
  从源头消除 close-to-close 的隔夜跳空未来函数（见 docs/project_audit §6/§8）。
- label 仅用于验证/训练，**绝不可作为 feature**。
- 超额 label = 个股收益 - 中证1000(idx_000852) 同窗口收益。
- 行业超额: 无行业映射数据，暂不生成。
- 单位: 小数收益（0.02 = 2%）。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DAILY_DIR = PROJECT / "data" / "daily"
DEFAULT_HORIZONS = [1, 3, 5, 10, 20]
INDEX_SYMBOL = "000852"  # 中证1000


def _open_to_open_labels(open_arr: np.ndarray, horizons: list[int]) -> dict[str, np.ndarray]:
    """给定按日期升序的 open 序列，返回各 horizon 的 open[T+1+h]/open[T+1]-1。"""
    n = len(open_arr)
    entry = np.full(n, np.nan)           # T+1 open
    entry[: n - 1] = open_arr[1:]
    out = {}
    for h in horizons:
        exit_ = np.full(n, np.nan)       # T+(1+h) open
        k = n - (1 + h)
        if k > 0:
            exit_[:k] = open_arr[1 + h : 1 + h + k]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"label_{h}d"] = exit_ / entry - 1.0
    return out


def _index_labels(start_date: str, end_date: str, horizons: list[int]) -> pd.DataFrame:
    """中证1000 指数的 open-to-open 收益，按 trade_date。"""
    p = DAILY_DIR / f"idx_{INDEX_SYMBOL}.parquet"
    if not p.exists():
        return pd.DataFrame()
    end_ts = pd.to_datetime(end_date)
    idx = pd.read_parquet(p)
    idx["date"] = pd.to_datetime(idx["date"])
    # 保留 end_date 之后的行用于末端远期出场价，最后再按信号日过滤输出
    idx = idx[idx["date"] >= start_date].sort_values("date").reset_index(drop=True)
    if idx.empty:
        return pd.DataFrame()
    labels = _open_to_open_labels(idx["open"].to_numpy(float), horizons)
    out = pd.DataFrame({"trade_date": idx["date"].to_numpy()})
    for h in horizons:
        out[f"idx_label_{h}d"] = labels[f"label_{h}d"]
    return out[out["trade_date"] <= end_ts]


def build_labels(
    codes: list[str], start_date: str, end_date: str,
    horizons: list[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """为给定股票池构建多周期 label + 指数超额 label。"""
    end_ts = pd.to_datetime(end_date)
    frames = []
    for code in codes:
        code = str(code).zfill(6)
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        # 保留 end_date 之后的行用于计算末端 T+1+h 远期出场价，避免年末 label 被截断为 NaN
        df = df[df["date"] >= start_date].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        labels = _open_to_open_labels(df["open"].to_numpy(float), horizons)
        rec = {"trade_date": df["date"].to_numpy(), "symbol": code}
        rec.update(labels)
        one = pd.DataFrame(rec)
        frames.append(one[one["trade_date"] <= end_ts])   # 输出仅保留窗口内信号日
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    idx_df = _index_labels(start_date, end_date, horizons)
    if not idx_df.empty:
        out = out.merge(idx_df, on="trade_date", how="left")
        for h in horizons:
            out[f"label_{h}d_excess_index"] = out[f"label_{h}d"] - out[f"idx_label_{h}d"]
        out = out.drop(columns=[f"idx_label_{h}d" for h in horizons])
    return out
