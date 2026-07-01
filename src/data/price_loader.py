"""
日线行情加载器
akshare stock_zh_a_daily() (Sina来源) + stock_zh_index_daily() (Sina来源)
后复权数据，本地Parquet缓存
带重试+递增延迟，避免API限流
"""
from __future__ import annotations
from pathlib import Path
import time
import random
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
import akshare as ak

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "daily"

# 请求间隔控制（秒），避免Sina API频率限制
_MIN_REQUEST_GAP = 0.3
_last_request_time = 0.0


def _to_sina_symbol(code: str) -> str:
    """将纯代码转为 stock_zh_a_daily 所需格式: 000001 -> sz000001, 600036 -> sh600036"""
    code = str(code).zfill(6)
    if code[0] in ('6', '9'):
        return f"sh{code}"
    else:
        return f"sz{code}"


def _rate_limit():
    """确保两次API调用之间有最小间隔"""
    global _last_request_time
    now = time.time()
    gap = now - _last_request_time
    if gap < _MIN_REQUEST_GAP:
        time.sleep(_MIN_REQUEST_GAP - gap)
    _last_request_time = time.time()


def _fetch_with_retry(fn, max_retries=3, base_delay=2.0):
    """带指数退避重试的API调用"""
    last_err = None
    for attempt in range(max_retries):
        try:
            _rate_limit()
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API调用失败 (attempt {attempt+1}), {delay:.1f}s后重试: {e}")
                time.sleep(delay)
    raise last_err


def load_stock_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "hfq",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    加载单只股票日线后复权数据

    symbol: 纯数字代码如 '000001', '600036'
    adjust: 'hfq'=后复权, ''=不复权, 'qfq'=前复权
        注意: hfq 返回价格单位为分(需/100), 不复权返回元
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_symbol = str(symbol)
    symbol = raw_symbol.zfill(6)
    cache_path = cache_dir / f"{symbol}.parquet"
    legacy_cache_path = cache_dir / f"{raw_symbol}.parquet"
    if not cache_path.exists() and legacy_cache_path.exists():
        cache_path = legacy_cache_path

    sina_sym = _to_sina_symbol(symbol)

    need_download = True
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if 'date' in cached.columns:
            cached["date"] = pd.to_datetime(cached["date"])
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            cached_window = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
            if not cached_window.empty:
                need_download = False

    if need_download:
        try:
            raw = _fetch_with_retry(
                lambda: ak.stock_zh_a_daily(
                    symbol=sina_sym,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust,
                )
            )
            if raw is None or raw.empty:
                logger.warning(f"{symbol} 无日线数据")
                if cache_path.exists():
                    return pd.read_parquet(cache_path)
                return pd.DataFrame()

            raw["date"] = pd.to_datetime(raw["date"])

            # 注意: akshare >=1.14 的 stock_zh_a_daily hfq 已直接返回元
            # 不再需要 /100 转换。旧版缓存（分单位）通过删除缓存重建来处理

            raw = raw.sort_values("date")

            if cache_path.exists():
                old = pd.read_parquet(cache_path)
                old["date"] = pd.to_datetime(old["date"])
                combined = pd.concat([old, raw], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined = combined.sort_values("date")
                combined.to_parquet(cache_path, index=False)
            else:
                raw.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.error(f"{symbol} 下载失败(已重试): {e}")
            if cache_path.exists():
                return pd.read_parquet(cache_path)
            return pd.DataFrame()

    df = pd.read_parquet(cache_path)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].reset_index(drop=True)


def _to_sina_index_symbol(code: str) -> str:
    """
    将指数代码转为 Sina 格式

    支持输入: '000852', '000852.SH', 'sh000852', 'sz399101'
    输出: 'sh000852', 'sz399101' 等
    """
    code = str(code).upper().replace('.SH', '').replace('.SZ', '').replace('SH', '').replace('SZ', '')
    code = code.zfill(6)
    if code.startswith('399') or code.startswith('399'):
        return f"sz{code}"
    else:
        return f"sh{code}"


def load_index_daily(
    symbol: str = "sh000852",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    加载指数日线（Sina来源，用于基准对比）

    symbol: 指数代码，支持多种格式
        'sh000852' / '000852' / '000852.SH' = 中证1000
        'sz399101' = 中小板综指
        'sh000001' = 上证综指
        'sh000300' = 沪深300
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_code = symbol.replace('sz', '').replace('sh', '').replace('.SH', '').replace('.SZ', '').zfill(6)
    cache_path = cache_dir / f"idx_{clean_code}.parquet"
    sina_sym = _to_sina_index_symbol(symbol)

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if 'date' in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    else:
        try:
            df = _fetch_with_retry(
                lambda: ak.stock_zh_index_daily(symbol=sina_sym)
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.error(f"指数 {sina_sym} 下载失败: {e}")
            return pd.DataFrame()

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].reset_index(drop=True)
