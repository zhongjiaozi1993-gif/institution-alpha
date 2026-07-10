"""ML-ready 宽表构建器。

把 feature（Alpha191）+ label + tradable flag + universe 成员合并为标准宽表:
    trade_date | symbol | in_Universe_A/B/C | feat_* | label_* | *_flag

无未来函数保证:
- feature 于 T 日收盘后可得（available_time = T 收盘后），只可用于 T+1 交易。
- label 于 T+1 open 起算（见 label_builder），只用于验证/训练。
- 二者对齐在同一 trade_date T，feature 早于 label 的可用时点，无泄漏。

后续阶段（Level-2 / 龙虎榜 / 北向）以相同键 (trade_date, symbol) 增量并入。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
SIGNAL_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191_full"
LABELS = PROJECT / "data" / "processed" / "labels" / "labels.parquet"
FLAGS = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"


def load_alpha191_features(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 30 个 Alpha191 信号 parquet 合并为宽表 + feature 元信息。

    返回 (wide_df[trade_date, symbol, feat_*], meta_df[feature, source, version, signal_id]).
    """
    files = sorted(SIGNAL_DIR.glob("signal*.parquet"))
    wide = None
    meta = []
    for fp in files:
        s = pd.read_parquet(fp)
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        s = s[(s["trade_date"] >= start_date) & (s["trade_date"] <= end_date)]
        sid = str(s["signal_id"].iloc[0]) if "signal_id" in s.columns else fp.stem
        feat = f"feat_{sid.lower()}"
        col = s[["trade_date", "stock_code", "signal_value"]].rename(
            columns={"stock_code": "symbol", "signal_value": feat})
        col["symbol"] = col["symbol"].astype(str).str.zfill(6)
        wide = col if wide is None else wide.merge(col, on=["trade_date", "symbol"], how="outer")
        meta.append({
            "feature": feat,
            "signal_id": sid,
            "signal_name": str(s["signal_name"].iloc[0]) if "signal_name" in s.columns else "",
            "source": "alpha191",
            "source_formula_id": str(s["source_formula_id"].iloc[0]) if "source_formula_id" in s.columns else "",
            "available_time": "T_close",
            "version": "v1",
        })
    if wide is None:
        return pd.DataFrame(), pd.DataFrame()
    return wide, pd.DataFrame(meta)


def build_ml_dataset(
    universe_symbols: dict[str, set], start_date: str, end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建 ML 宽表。spine = labels（universe 并集的 date-symbol）。"""
    labels = pd.read_parquet(LABELS)
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["symbol"] = labels["symbol"].astype(str).str.zfill(6)

    flags = pd.read_parquet(FLAGS)
    flags["trade_date"] = pd.to_datetime(flags["trade_date"])
    flags["symbol"] = flags["symbol"].astype(str).str.zfill(6)

    feats, meta = load_alpha191_features(start_date, end_date)

    df = labels.merge(feats, on=["trade_date", "symbol"], how="left")
    df = df.merge(flags, on=["trade_date", "symbol"], how="left")

    # universe 成员标记
    for uid, syms in universe_symbols.items():
        df[f"in_{uid}"] = df["symbol"].isin(syms)

    # 列排序: 键 → universe → feat → label → flag
    key_cols = ["trade_date", "symbol"]
    uni_cols = [f"in_{u}" for u in universe_symbols]
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    label_cols = [c for c in df.columns if c.startswith("label_")]
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    ordered = key_cols + uni_cols + feat_cols + label_cols + flag_cols
    df = df[[c for c in ordered if c in df.columns]].sort_values(key_cols).reset_index(drop=True)
    return df, meta
