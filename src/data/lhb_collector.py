"""
龙虎榜数据采集器
使用 akshare 新浪来源API（东方财富API在新版akshare中不稳定）
提供三层数据：每日上榜股票列表、机构席位买卖明细、营业部排名
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
import akshare as ak

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "lhb"


def download_lhb_stocks_daily(date: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """
    下载单日龙虎榜上榜股票列表（新浪来源）
    Columns: 序号, 股票代码, 股票名称, 收盘价, 对应值, 成交量, 成交额, 指标
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"lhb_stocks_{date}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path, dtype={"股票代码": str})

    df = ak.stock_lhb_detail_daily_sina(date=date)
    if df is None or df.empty:
        logger.warning(f"龙虎榜 {date} 无上榜股票")
        return pd.DataFrame()

    df["上榜日"] = date
    df.to_csv(cache_path, index=False)
    logger.info(f"龙虎榜 {date}: {len(df)} 只上榜股票")
    return df


def download_lhb_jgmx(
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    max_pages: int = 20,
) -> pd.DataFrame:
    """
    下载机构席位买卖明细（新浪来源，支持翻页获取更多历史数据）

    Sina默认只显示最近一周数据（约8页/304条），翻页可回溯更早记录。
    Columns: 股票代码, 股票名称, 交易日期, 机构席位买入额(万), 机构席位卖出额(万), 类型
    """
    import requests
    from io import StringIO

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "lhb_jgmx.parquet"

    if not force_refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty:
            return cached

    all_frames = []
    for p in range(1, max_pages + 1):
        try:
            url = f"https://vip.stock.finance.sina.com.cn/q/go.php/vLHBData/kind/jgmx/index.phtml?p={p}"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            tables = pd.read_html(StringIO(r.text))
            if not tables or len(tables[0]) == 0:
                break
            df = tables[0]
            if "交易日期" not in df.columns:
                break
            all_frames.append(df)
        except Exception as e:
            logger.warning(f"LHB翻页 p={p} 失败: {e}")
            break

    if not all_frames:
        logger.error("未获取到任何机构席位明细")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(subset=["股票代码", "交易日期", "类型"], keep="first")
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    df = df.rename(columns={
        "机构席位买入额(万)": "机构席位买入额",
        "机构席位卖出额(万)": "机构席位卖出额",
    })
    df.to_parquet(cache_path, index=False)
    logger.info(f"机构席位明细: {len(df)} 条记录, {df['交易日期'].min().date()} ~ {df['交易日期'].max().date()}")
    return df


def download_lhb_seat_stats(cache_dir: Path | None = None) -> pd.DataFrame:
    """
    下载营业部排名统计（新浪来源，累积统计）
    Columns: 营业部名称, 上榜次数, 累积购买额, 买入席位数, 累积卖出额, 卖出席位数, 买入前三股票
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "lhb_seat_stats.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = ak.stock_lhb_yytj_sina()
    if df is None or df.empty:
        return pd.DataFrame()

    df.to_parquet(cache_path, index=False)
    logger.info(f"营业部排名: {len(df)} 条记录")
    return df


def download_lhb_seat_detail(
    date: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    下载单日龙虎榜完整席位明细（东方财富来源，含每个营业部的买卖金额）

    流程: stock_lhb_detail_daily_sina(date) 获取上榜股票列表
         → 对每只股票调用 stock_lhb_stock_detail_em() 获取席位明细

    Columns: stock_code, stock_name, lhb_date, seat_name, buy_amount, sell_amount, net_amount
    """
    import time as _time
    import random as _random

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"lhb_seat_detail_{date}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Step 1: 获取当日上榜股票列表
    stocks = download_lhb_stocks_daily(date, cache_dir)
    if stocks.empty:
        return pd.DataFrame()

    all_records = []
    codes = stocks["股票代码"].astype(str).unique()

    for i, code in enumerate(codes):
        # 请求间隔，避免东方财富限流
        if i > 0:
            _time.sleep(0.3 + _random.uniform(0, 0.2))

        for flag in ["买入", "卖出"]:
            try:
                detail = ak.stock_lhb_stock_detail_em(
                    symbol=code, date=date.replace("-", ""), flag=flag
                )
                if detail is None or detail.empty:
                    continue

                for _, row in detail.iterrows():
                    seat = row.get("交易营业部名称", "")
                    buy_amt = row.get("买入金额", 0) or 0
                    sell_amt = row.get("卖出金额", 0) or 0
                    if buy_amt == 0 and sell_amt == 0:
                        continue

                    all_records.append({
                        "stock_code": str(code),
                        "stock_name": stocks[stocks["股票代码"] == code]["股票名称"].values[0]
                            if len(stocks[stocks["股票代码"] == code]) > 0 else "",
                        "lhb_date": date,
                        "seat_name": str(seat),
                        "buy_amount": float(buy_amt),
                        "sell_amount": float(sell_amt),
                        "net_amount": float(buy_amt) - float(sell_amt),
                    })
            except Exception as e:
                logger.warning(f"席位明细 {code}/{flag} 失败: {e}")
                continue

    if not all_records:
        logger.warning(f"龙虎榜 {date}: 无席位明细")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["lhb_date"] = pd.to_datetime(df["lhb_date"])
    df.to_parquet(cache_path, index=False)
    logger.info(f"龙虎榜席位明细 {date}: {len(df)} 条, {df['seat_name'].nunique()} 个营业部")
    return df


def download_lhb_seat_detail_range(
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    下载日期范围内的完整席位明细

    在 start_date~end_date 的每个交易日批量获取席位级龙虎榜数据
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dates = pd.bdate_range(start=start_date, end=end_date)
    frames = []
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        df = download_lhb_seat_detail(d_str, cache_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_trade_records(
    jgmx_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    从机构买卖明细构建标准化的交易记录（用于Alpha归因）

    jgmx_df: 来自 download_lhb_jgmx() 的机构席位明细
        列: 股票代码, 股票名称, 交易日期, 机构席位买入额, 机构席位卖出额, 类型

    Returns: [stock_code, lhb_date, seat_name, buy_amount, sell_amount]
    """
    if jgmx_df.empty:
        return pd.DataFrame()

    df = jgmx_df.copy()
    df["交易日期"] = pd.to_datetime(df["交易日期"])

    if start_date:
        mask = df["交易日期"] >= start_date
        df = df[mask]
    if end_date:
        mask = df["交易日期"] <= end_date
        df = df[mask]

    records = df.rename(columns={
        "股票代码": "stock_code",
        "股票名称": "stock_name",
        "交易日期": "lhb_date",
        "机构席位买入额": "buy_amount",
        "机构席位卖出额": "sell_amount",
    })

    # Sina jgmx 金额单位为万元，转为元以统一口径
    records["buy_amount"] = pd.to_numeric(records["buy_amount"], errors="coerce").fillna(0) * 10000
    records["sell_amount"] = pd.to_numeric(records["sell_amount"], errors="coerce").fillna(0) * 10000

    records["seat_name"] = "机构专用"  # jgmx only has institution aggregate
    records["stock_code"] = records["stock_code"].astype(str)
    records = records[["stock_code", "stock_name", "lhb_date", "seat_name",
                        "buy_amount", "sell_amount", "类型"]]

    return records.reset_index(drop=True)


def download_lhb_range(start_date: str, end_date: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """
    下载日期范围内上榜股票列表（兼容旧接口）
    批量下载 + 合并
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dates = pd.bdate_range(start=start_date, end=end_date)
    frames = []
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        df = download_lhb_stocks_daily(d_str, cache_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_lhb_range(start_date: str, end_date: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """
    下载日期范围内上榜股票列表（兼容旧接口）
    批量下载 + 合并
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dates = pd.bdate_range(start=start_date, end=end_date)
    frames = []
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        df = download_lhb_stocks_daily(d_str, cache_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
