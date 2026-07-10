"""Universe Registry: 构建与加载 Universe_A / Universe_B / Universe_C。

设计原则:
- 核心逻辑在此模块，可执行编排在 scripts/build_universe.py。
- 每个 universe 落地为静态成员表 parquet（symbol 级 + 统计列）。
- market cap 口径：**不可用后复权 close × 股本**（会得到荒谬值），
  须用真实价 ≈ amount/volume(VWAP) × outstanding_share。见 docs/project_audit §10。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DAILY_DIR = PROJECT / "data" / "daily"
UNIVERSE_DIR = PROJECT / "data" / "processed" / "universe"
REGISTRY_CSV = PROJECT / "signal_zoo" / "registry" / "universe_registry.csv"

STAT_COLS = [
    "symbol", "n_rows", "n_trading_days", "first_date", "last_date",
    "median_amount", "avg_turnover", "market_cap_est",
]


def compute_stock_stats(code: str, start_date: str, end_date: str) -> dict | None:
    """单只股票在 [start, end] 窗口内的基础统计。返回 None 表示无日线数据。"""
    code = str(code).zfill(6)
    p = DAILY_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" not in df.columns or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date")
    if df.empty:
        return None

    traded = df[df["volume"] > 0]
    n_traded = len(traded)
    real_price = (traded["amount"] / traded["volume"]).replace([np.inf, -np.inf], np.nan)
    last_price = real_price.dropna().iloc[-1] if real_price.notna().any() else np.nan
    shares = df["outstanding_share"].dropna()
    last_shares = shares.iloc[-1] if len(shares) else np.nan
    mktcap = (last_price * last_shares
              if pd.notna(last_price) and pd.notna(last_shares) else np.nan)

    return {
        "symbol": code,
        "n_rows": len(df),
        "n_trading_days": n_traded,
        "first_date": df["date"].min(),
        "last_date": df["date"].max(),
        "median_amount": float(traded["amount"].median()) if n_traded else 0.0,
        "avg_turnover": float(traded["turnover"].mean()) if n_traded else np.nan,
        "market_cap_est": float(mktcap) if pd.notna(mktcap) else np.nan,
    }


def _exclude_reason(stat: dict, filters: dict) -> str | None:
    """套用通用过滤，返回剔除原因；None 表示保留。ST 无数据源，无法执行。"""
    if stat is None:
        return "no_daily_data"
    if stat["n_rows"] < filters["new_ipo_days"]:
        return "new_stock"
    if stat["n_rows"] < filters["min_trading_days"]:
        return "insufficient_data"
    if stat["median_amount"] < filters["min_median_amount"]:
        return "low_liquidity"
    return None


def build_membership(
    candidate_codes: list[str], start_date: str, end_date: str, filters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对候选池套用过滤，返回 (included_df, excluded_df)。"""
    included, excluded = [], []
    for code in candidate_codes:
        stat = compute_stock_stats(code, start_date, end_date)
        reason = _exclude_reason(stat, filters)
        if reason is None:
            included.append(stat)
        else:
            row = {"symbol": str(code).zfill(6), "exclude_reason": reason}
            if stat is not None:
                row["n_rows"] = stat["n_rows"]
                row["median_amount"] = stat["median_amount"]
            excluded.append(row)
    inc_df = pd.DataFrame(included, columns=STAT_COLS) if included else pd.DataFrame(columns=STAT_COLS)
    exc_df = pd.DataFrame(excluded) if excluded else pd.DataFrame(columns=["symbol", "exclude_reason"])
    return inc_df, exc_df


def save_universe(members: pd.DataFrame, universe_id: str, name: str, purpose: str) -> Path:
    """落地成员表 parquet（附 universe 元信息列）。"""
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    out = members.copy()
    out.insert(1, "universe_id", universe_id)
    out.insert(2, "universe_name", name)
    out.insert(3, "purpose", purpose)
    path = UNIVERSE_DIR / f"{universe_id}.parquet"
    out.to_parquet(path, index=False)
    return path


def load_universe(universe_id: str) -> list[str]:
    """加载某 universe 的股票代码列表。"""
    path = UNIVERSE_DIR / f"{universe_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Universe not built: {path}")
    df = pd.read_parquet(path)
    return sorted(df["symbol"].astype(str).str.zfill(6).unique())


def load_universe_table(universe_id: str) -> pd.DataFrame:
    return pd.read_parquet(UNIVERSE_DIR / f"{universe_id}.parquet")


def update_registry(rows: list[dict]) -> None:
    """重写 universe_registry.csv 为 A/B/C 三行。"""
    cols = ["universe_id", "universe_name", "data_requirement", "source",
            "stock_count", "start_date", "end_date", "status", "notes"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(REGISTRY_CSV, index=False)
