"""Level-2 日频特征生产器（Phase 5, v1）。

把已有的 Level-2 读取器（src/data/level2_reader）与 DBSCAN 拆单检测器
（src/cluster/split_detector）改造成**日频特征产出**：每只股票每个交易日一行。

无未来函数保证
--------------
所有特征仅使用 **T 日当日逐笔**（逐笔委托 + 逐笔成交），于 **T 日收盘后**可得
（available_time = T_close）。不使用 T 日之后任何信息，也不做跨日滚动窗口（避免
Level-2 覆盖稀疏造成的窗口泄漏/缺口）。label 从 T+1 开盘起算（label_builder），
与特征可用时点严格错开，无泄漏。

特征分组（v1 共 35 个，命名前缀 l2_）
  flow(7)      : 成交额/量、笔数、主动买卖占比、净主动买入强度
  intraday(3)  : 日内涨跌、收盘位置、收盘相对 VWAP 偏离
  session(2)   : 早盘/尾盘净主动买入强度
  large(12)    : 超大/大单买卖（亿）、净额、强度、占比 + 委托口径大单
  cluster(11)  : DBSCAN 拆单集群数/金额/买入强度/拆单指纹（HHI/笔数/追价/时长）

单位口径（沿用 level2_reader）
  成交价格 = 元 × 10000；成交金额 = 成交价格 × 成交数量 → /1e12=亿元, /1e8=万元。
  委托金额 = 委托价格 × 委托数量，同口径。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.data import level2_reader as l2
from src.cluster.split_detector import detect_institution_operations

PROJECT = Path(__file__).resolve().parent.parent.parent
SINGLE_STOCK = PROJECT / "data" / "single_stock"

# 特征版本 + DBSCAN 参数（写入 metadata / run_manifest，防止版本漂移）。
FEATURE_VERSION = "v1"
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 5
DBSCAN_MIN_TOTAL_WAN = 100.0

TRADE_FILE = "逐笔成交.csv"
ORDER_FILE = "逐笔委托.csv"

# 覆盖审计的最终列（每个 symbol-day 恰好一行）。
AUDIT_COLUMNS = [
    "symbol", "day", "trade_date", "selected_layout",
    "has_trade_file", "has_order_file", "status", "reason",
    "n_trades", "n_orders", "feature_version",
]


class LayoutResolution(NamedTuple):
    selected_dir: Path | None
    layout_type: str  # wind_subdir / flat_day_dir / no_supported_layout
    has_trade_file: bool
    has_order_file: bool


def resolve_level2_day_dir(code: str, day_dir: str | Path) -> LayoutResolution:
    """判定某股票某交易日的逐笔文件所在结构，依据是**实际文件是否存在**（不只是目录）。

    结构 A wind_subdir: {day_dir}/{code}.SZ/逐笔成交.csv
    结构 B flat_day_dir: {day_dir}/逐笔成交.csv

    两者都有 逐笔成交.csv 时优先 wind_subdir（深度池原始结构）。都没有 → no_supported_layout。
    """
    day_dir = Path(day_dir)
    wind_dir = day_dir / l2._build_wind_code(code)
    if (wind_dir / TRADE_FILE).exists():
        return LayoutResolution(wind_dir, "wind_subdir", True, (wind_dir / ORDER_FILE).exists())
    if (day_dir / TRADE_FILE).exists():
        return LayoutResolution(day_dir, "flat_day_dir", True, (day_dir / ORDER_FILE).exists())
    return LayoutResolution(None, "no_supported_layout", False, False)

# 特征名 → 中文说明（既驱动元信息，也用于防止代码与元信息漂移）。
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # ---- flow（逐笔成交，全量）----
    "l2_amount_yi": "当日成交额（亿元）",
    "l2_volume_wan": "当日成交量（万股）",
    "l2_trade_count": "逐笔成交笔数",
    "l2_avg_trade_amt_wan": "单笔平均成交额（万元）",
    "l2_active_buy_ratio": "主动买成交额占比（BS=B / 总额）",
    "l2_active_sell_ratio": "主动卖成交额占比（BS=S / 总额）",
    "l2_net_active_ratio": "净主动买入强度 (买-卖)/总额",
    # ---- intraday（日内，T 收盘可得）----
    "l2_intraday_ret": "日内涨跌幅 (close-open)/open %",
    "l2_close_pos": "收盘在日内区间位置 (close-low)/(high-low)",
    "l2_vwap_close_dev": "收盘相对 VWAP 偏离 %",
    # ---- session（时段）----
    "l2_early_net_ratio": "早盘 09:30-10:00 净主动买入 / 总额",
    "l2_late_net_ratio": "尾盘 14:30-15:00 净主动买入 / 总额",
    # ---- large orders（委托-成交匹配后按规模分层）----
    "l2_super_buy_yi": "超大单买入额（亿, ≥100万或≥50万股）",
    "l2_super_sell_yi": "超大单卖出额（亿）",
    "l2_super_net_yi": "超大单净买入（亿）",
    "l2_big_buy_yi": "大单买入额（亿, ≥20万或≥10万股, 不含超大）",
    "l2_big_sell_yi": "大单卖出额（亿）",
    "l2_big_net_yi": "大单净买入（亿）",
    "l2_super_buy_ratio": "超大单买入 / 当日成交额",
    "l2_big_net_ratio": "(超大+大)净买入 / 当日成交额",
    "l2_large_share": "(超大+大)买卖合计 / 当日成交额（大单参与度）",
    "l2_order_count": "成交委托数（匹配后）",
    "l2_buy_order_ratio": "买方向委托占比（委托代码=B）",
    "l2_avg_order_wan": "平均委托金额（万元）",
    # ---- DBSCAN 拆单集群（匿名机构操作指纹）----
    "l2_cluster_count": "拆单集群总数",
    "l2_buy_cluster_count": "买入集群数",
    "l2_sell_cluster_count": "卖出集群数",
    "l2_cluster_buy_wan": "买入集群总额（万元）",
    "l2_cluster_sell_wan": "卖出集群总额（万元）",
    "l2_cluster_net_wan": "集群净买入（万元）",
    "l2_cluster_buy_intensity": "机构买入强度 = 买入集群额 / 当日成交额",
    "l2_max_cluster_wan": "最大单集群金额（万元）",
    "l2_avg_cluster_hhi": "集群平均拆单集中度 HHI（1=集中,→1/n=均匀）",
    "l2_avg_cluster_orders": "集群平均拆单笔数",
    "l2_avg_cluster_vwap_dev": "集群平均成交价相对 VWAP 偏离 %（正=追价）",
}

FEATURE_NAMES: list[str] = list(FEATURE_DESCRIPTIONS.keys())

_LARGE_ZERO_KEYS = [
    "l2_super_buy_yi", "l2_super_sell_yi", "l2_super_net_yi",
    "l2_big_buy_yi", "l2_big_sell_yi", "l2_big_net_yi",
    "l2_super_buy_ratio", "l2_big_net_ratio", "l2_large_share",
    "l2_buy_order_ratio", "l2_avg_order_wan",
]
_CLUSTER_ZERO_KEYS = [
    "l2_cluster_count", "l2_buy_cluster_count", "l2_sell_cluster_count",
    "l2_cluster_buy_wan", "l2_cluster_sell_wan", "l2_cluster_net_wan",
    "l2_cluster_buy_intensity", "l2_max_cluster_wan",
    "l2_avg_cluster_hhi", "l2_avg_cluster_orders", "l2_avg_cluster_vwap_dev",
]


def _session_net_ratio(cj: pd.DataFrame, total_amt: float, start_hhmm: str, end_hhmm: str) -> float:
    """某时段净主动买入额 / 全日总额。时间字段为 HHMMSSmmm，取前 4 位比较。"""
    if cj.empty or total_amt <= 0 or "BS标志" not in cj.columns:
        return 0.0
    hhmm = cj["时间"].astype(str).str.zfill(9).str[:4]
    sub = cj[(hhmm >= start_hhmm) & (hhmm < end_hhmm)]
    if sub.empty:
        return 0.0
    buy = sub.loc[sub["BS标志"] == "B", "成交金额"].sum()
    sell = sub.loc[sub["BS标志"] == "S", "成交金额"].sum()
    return float((buy - sell) / total_amt)


def compute_day_features(wt: pd.DataFrame | None, cj: pd.DataFrame | None) -> dict | None:
    """单只股票单日 → 特征 dict（35 个）。cj 为空返回 None（无有效成交）。"""
    if cj is None or cj.empty:
        return None
    feats: dict[str, float] = {}

    # ---- flow ----
    total_amt = float(cj["成交金额"].sum())        # raw（元×10000×股）
    total_amt_yi = total_amt / 1e12
    total_amt_wan = total_amt / 1e8
    total_vol = float(cj["成交数量"].sum())
    ohlcv = l2.compute_daily_ohlcv(cj)
    buy_amt = float(cj.loc[cj["BS标志"] == "B", "成交金额"].sum()) if "BS标志" in cj.columns else 0.0
    sell_amt = float(cj.loc[cj["BS标志"] == "S", "成交金额"].sum()) if "BS标志" in cj.columns else 0.0

    feats["l2_amount_yi"] = round(total_amt_yi, 4)
    feats["l2_volume_wan"] = round(float(ohlcv["volume"]), 2)
    feats["l2_trade_count"] = int(len(cj))
    feats["l2_avg_trade_amt_wan"] = round(total_amt_wan / max(len(cj), 1), 4)
    feats["l2_active_buy_ratio"] = round(buy_amt / total_amt, 4) if total_amt > 0 else 0.0
    feats["l2_active_sell_ratio"] = round(sell_amt / total_amt, 4) if total_amt > 0 else 0.0
    feats["l2_net_active_ratio"] = round((buy_amt - sell_amt) / total_amt, 4) if total_amt > 0 else 0.0

    # ---- intraday ----
    o, c, hi, lo = float(ohlcv["open"]), float(ohlcv["close"]), float(ohlcv["high"]), float(ohlcv["low"])
    feats["l2_intraday_ret"] = round((c - o) / o * 100, 4) if o > 0 else 0.0
    feats["l2_close_pos"] = round((c - lo) / (hi - lo), 4) if hi > lo else 0.5
    vwap = (total_amt / total_vol / 10000) if total_vol > 0 else c   # 元
    feats["l2_vwap_close_dev"] = round((c - vwap) / vwap * 100, 4) if vwap > 0 else 0.0

    # ---- session ----
    feats["l2_early_net_ratio"] = round(_session_net_ratio(cj, total_amt, "0930", "1000"), 4)
    feats["l2_late_net_ratio"] = round(_session_net_ratio(cj, total_amt, "1430", "1500"), 4)

    # ---- large orders（需委托-成交匹配）----
    wtcj = l2.match_orders_to_trades(wt, cj) if wt is not None and not wt.empty else pd.DataFrame()
    if not wtcj.empty:
        bs = l2.compute_big_order_summary(wtcj)  # 亿元；big 为“大单但非超大”层
        sb, ss = bs["super_buy"], bs["super_sell"]
        bb, bsl = bs["big_buy"], bs["big_sell"]
        feats["l2_super_buy_yi"] = round(float(sb), 4)
        feats["l2_super_sell_yi"] = round(float(ss), 4)
        feats["l2_super_net_yi"] = round(float(sb - ss), 4)
        feats["l2_big_buy_yi"] = round(float(bb), 4)
        feats["l2_big_sell_yi"] = round(float(bsl), 4)
        feats["l2_big_net_yi"] = round(float(bb - bsl), 4)
        feats["l2_super_buy_ratio"] = round(sb / total_amt_yi, 4) if total_amt_yi > 0 else 0.0
        feats["l2_big_net_ratio"] = round(((sb + bb) - (ss + bsl)) / total_amt_yi, 4) if total_amt_yi > 0 else 0.0
        feats["l2_large_share"] = round((sb + ss + bb + bsl) / total_amt_yi, 4) if total_amt_yi > 0 else 0.0
        feats["l2_order_count"] = int(len(wtcj))
        feats["l2_buy_order_ratio"] = round(float((wtcj["委托代码"] == "B").mean()), 4)
        feats["l2_avg_order_wan"] = round(float(wtcj["委托金额"].mean()) / 1e8, 4)
    else:
        for k in _LARGE_ZERO_KEYS:
            feats[k] = 0.0
        feats["l2_order_count"] = int(len(wt)) if wt is not None else 0

    # ---- DBSCAN 拆单集群 ----
    ops = detect_institution_operations(
        wtcj, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
        min_total_amount_wan=DBSCAN_MIN_TOTAL_WAN,
    ) if not wtcj.empty else []
    if ops:
        buy_ops = [op for op in ops if op["direction"] == "BUY"]
        sell_ops = [op for op in ops if op["direction"] == "SELL"]
        buy_wan = sum(op["total_amount_wan"] for op in buy_ops)
        sell_wan = sum(op["total_amount_wan"] for op in sell_ops)
        feats["l2_cluster_count"] = len(ops)
        feats["l2_buy_cluster_count"] = len(buy_ops)
        feats["l2_sell_cluster_count"] = len(sell_ops)
        feats["l2_cluster_buy_wan"] = round(buy_wan, 2)
        feats["l2_cluster_sell_wan"] = round(sell_wan, 2)
        feats["l2_cluster_net_wan"] = round(buy_wan - sell_wan, 2)
        feats["l2_cluster_buy_intensity"] = round(buy_wan / total_amt_wan, 4) if total_amt_wan > 0 else 0.0
        feats["l2_max_cluster_wan"] = round(max(op["total_amount_wan"] for op in ops), 2)
        feats["l2_avg_cluster_hhi"] = round(float(np.mean([op["order_hhi"] for op in ops])), 4)
        feats["l2_avg_cluster_orders"] = round(float(np.mean([op["order_count"] for op in ops])), 2)
        feats["l2_avg_cluster_vwap_dev"] = round(float(np.mean([op["vwap_deviation_pct"] for op in ops])), 4)
    else:
        for k in _CLUSTER_ZERO_KEYS:
            feats[k] = 0.0

    return feats


def _audit_row(code: str, day: str, trade_date, res: LayoutResolution) -> dict:
    return {
        "symbol": code, "day": day, "trade_date": trade_date,
        "selected_layout": res.layout_type,
        "has_trade_file": res.has_trade_file,
        "has_order_file": res.has_order_file,
        "status": None, "reason": "",
        "n_trades": 0, "n_orders": 0,
        "feature_version": FEATURE_VERSION,
    }


def build_stock_features(
    code: str,
    single_stock_root: str | Path | None = None,
    start_day: str | None = None,
    end_day: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """遍历某股票 {root}/{code}/raw/{YYYYMMDD}/ 的所有交易日 → (特征宽表, 审计行列表)。

    - 支持 wind_subdir 与 flat_day_dir 两种结构（resolve_level2_day_dir 按实际文件判定）。
    - **每个扫描到的 symbol-day 都产出恰好一行审计**，绝不静默 continue。
      status ∈ ok / no_supported_layout / empty_trades / decode_error / other_error。
    - start_day / end_day 为 'YYYYMMDD'（含端点）过滤，避免读取窗口外文件。
    - 所有特征仅用 T 日当日逐笔，无跨日状态、遍历顺序不影响结果。
    """
    root = Path(single_stock_root) if single_stock_root is not None else SINGLE_STOCK
    base = root / code / "raw"
    cols = ["trade_date", "symbol", "layout_type"] + FEATURE_NAMES
    audit_rows: list[dict] = []
    rows: list[dict] = []
    if not base.exists():
        return pd.DataFrame(columns=cols), audit_rows

    for day in sorted(d for d in os.listdir(base) if len(d) == 8 and d.isdigit()):
        if start_day is not None and day < start_day:
            continue
        if end_day is not None and day > end_day:
            continue
        day_dir = base / day
        trade_date = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}")
        res = resolve_level2_day_dir(code, day_dir)
        audit = _audit_row(code, day, trade_date, res)

        if res.layout_type == "no_supported_layout":
            audit["status"] = "no_supported_layout"
            audit["reason"] = "no 逐笔成交.csv in wind_subdir or flat_day_dir"
            audit_rows.append(audit)
            continue

        try:
            data = l2.read_level2_stock_dir(res.selected_dir)
            cj = data.get("逐笔成交")
            wt = data.get("逐笔委托")
            audit["n_trades"] = int(len(cj)) if cj is not None else 0
            audit["n_orders"] = int(len(wt)) if wt is not None else 0
            if cj is None or cj.empty:
                audit["status"] = "empty_trades"
                audit["reason"] = "逐笔成交 为空（无有效成交/全撤单/价格<=0）"
                audit_rows.append(audit)
                continue
            feats = compute_day_features(wt, cj)
        except UnicodeDecodeError as e:
            audit["status"] = "decode_error"
            audit["reason"] = f"{type(e).__name__}: {e}"[:300]
            audit_rows.append(audit)
            continue
        except Exception as e:  # noqa: BLE001 — 任何其它异常都必须进入审计，不得静默
            audit["status"] = "other_error"
            audit["reason"] = f"{type(e).__name__}: {e}"[:300]
            audit_rows.append(audit)
            continue

        if feats is None:
            audit["status"] = "empty_trades"
            audit["reason"] = "compute_day_features 返回 None（无有效成交）"
            audit_rows.append(audit)
            continue

        feats["trade_date"] = trade_date
        feats["symbol"] = code
        feats["layout_type"] = res.layout_type
        rows.append(feats)
        audit["status"] = "ok"
        audit_rows.append(audit)

    if not rows:
        return pd.DataFrame(columns=cols), audit_rows
    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df[cols], audit_rows


def build_all_features(
    codes: list[str],
    single_stock_root: str | Path | None = None,
    start_day: str | None = None,
    end_day: str | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对多只股票逐一构建并纵向拼接。返回 (特征宽表, 覆盖审计表)。"""
    cols = ["trade_date", "symbol", "layout_type"] + FEATURE_NAMES
    frames = []
    audit_all: list[dict] = []
    for i, code in enumerate(codes, 1):
        df, audit_rows = build_stock_features(
            code, single_stock_root=single_stock_root, start_day=start_day, end_day=end_day)
        audit_all.extend(audit_rows)
        if not df.empty:
            frames.append(df)
        if progress and (i % 20 == 0 or i == len(codes)):
            ok = sum(1 for a in audit_all if a["status"] == "ok")
            print(f"  [{i}/{len(codes)}] {code}: feat_rows={sum(len(f) for f in frames)} "
                  f"audit_days={len(audit_all)} ok={ok}", flush=True)

    audit_df = audit_frame(audit_all)
    if not frames:
        return pd.DataFrame(columns=cols), audit_df
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out = out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return out, audit_df


def audit_frame(audit_rows: list[dict]) -> pd.DataFrame:
    """审计行列表 → 规范化 DataFrame（含 month 便于报告）。"""
    if not audit_rows:
        df = pd.DataFrame(columns=AUDIT_COLUMNS)
        df["month"] = pd.Series(dtype="object")
        return df
    df = pd.DataFrame(audit_rows)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["month"] = df["trade_date"].dt.strftime("%Y-%m")
    df = df[AUDIT_COLUMNS + ["month"]]
    return df.sort_values(["symbol", "day"]).reset_index(drop=True)



def feature_metadata() -> pd.DataFrame:
    """特征元信息表：feature/description/group/source/available_time/version。"""
    def _group(name: str) -> str:
        if name in ("l2_amount_yi", "l2_volume_wan", "l2_trade_count", "l2_avg_trade_amt_wan",
                    "l2_active_buy_ratio", "l2_active_sell_ratio", "l2_net_active_ratio"):
            return "flow"
        if name in ("l2_intraday_ret", "l2_close_pos", "l2_vwap_close_dev"):
            return "intraday"
        if name in ("l2_early_net_ratio", "l2_late_net_ratio"):
            return "session"
        if name.startswith("l2_cluster") or name.startswith("l2_buy_cluster") \
                or name.startswith("l2_sell_cluster") or name.startswith("l2_max_cluster") \
                or name.startswith("l2_avg_cluster"):
            return "cluster"
        return "large"

    return pd.DataFrame([
        {"feature": n, "description": d, "group": _group(n),
         "source": "level2", "available_time": "T_close",
         "feature_version": FEATURE_VERSION,
         "dbscan_eps": DBSCAN_EPS,
         "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
         "dbscan_min_total_wan": DBSCAN_MIN_TOTAL_WAN}
        for n, d in FEATURE_DESCRIPTIONS.items()
    ])
