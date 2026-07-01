"""
Level-2逐笔数据读取器
基于Wind来源GB18030编码CSV，解析逐笔委托/逐笔成交/行情快照

参考: 各代码的意义.docx, l2_read.py (用户提供的已验证代码)

沪深交易所字段差异:
  沪市逐笔委托 — 委托类型: A=正常委托, D=撤单; 委托代码: I=竞价开始, O=竞价结束, J=连续结束, C=收盘, B=买, S=卖
  深市逐笔委托 — 委托类型: 0=限价, 1=市价, U=本方最优; 委托代码: B=买, S=卖
  沪市逐笔成交 — BS标志: B=主动买, S=主动卖 (成交代码无撤单)
  深市逐笔成交 — 成交代码: C=撤单, 0=成交; BS标志: B=主动买, S=主动卖
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# dtype 定义 — 匹配实际 CSV 列顺序（含末尾空列）
# ═══════════════════════════════════════════════════════════════

WTDATA_TYPES = {
    '万得代码': str, '交易所代码': str, '自然日': str, '时间': str,
    '委托编号': str, '交易所委托号': str, '委托类型': str, '委托代码': str,
    '委托价格': int, '委托数量': float, 'Unnamed: 10': str,
}

CJDATA_TYPES = {
    '万得代码': str, '交易所代码': str, '自然日': str, '时间': str,
    '成交编号': str, '成交代码': str, '委托代码': str, 'BS标志': str,
    '成交价格': int, '成交数量': float, '叫卖序号': str, '叫买序号': str,
    'Unnamed: 12': str,
}

HQDATA_TYPES = {
    '万得代码': str, '交易所代码': str, '自然日': str, '时间': str,
    '成交价': int, '成交量': float, '成交额': float, '成交笔数': int,
    'IOPV': str, '成交标志': str, 'BS标志': str, '当日累计成交量': float,
    '当日成交额': int, '最高价': int, '最低价': int, '开盘价': int,
    '前收盘': int, '申卖价1': int, '申卖价2': int, '申卖价3': int,
    '申卖价4': int, '申卖价5': int, '申卖价6': int, '申卖价7': int,
    '申卖价8': int, '申卖价9': int, '申卖价10': int,
    '申卖量1': float, '申卖量2': float, '申卖量3': float,
    '申卖量4': float, '申卖量5': float, '申卖量6': float,
    '申卖量7': float, '申卖量8': float, '申卖量9': float, '申卖量10': float,
    '申买价1': int, '申买价2': int, '申买价3': int, '申买价4': int,
    '申买价5': int, '申买价6': int, '申买价7': int, '申买价8': int,
    '申买价9': int, '申买价10': int,
    '申买量1': float, '申买量2': float, '申买量3': float,
    '申买量4': float, '申买量5': float, '申买量6': float,
    '申买量7': float, '申买量8': float, '申买量9': float, '申买量10': float,
    '加权平均叫卖价': int, '加权平均叫买价': int,
    '叫卖总量': float, '叫买总量': float, '不加权指数': int,
    '品种总数': int, '上涨品种数': int, '下跌品种数': int, '持平品种数': int,
    'Unnamed: 66': str,
}

# ═══════════════════════════════════════════════════════════════
# 大单/超大单阈值（金额单位: 分, 数量单位: 股）
# ═══════════════════════════════════════════════════════════════
SUPER_AMOUNT = 100 * 10000 * 10000    # 100万元 (以分为单位)
SUPER_VOLUME = 50 * 10000             # 50万股 (以股为单位)
BIG_AMOUNT = 20 * 10000 * 10000       # 20万元
BIG_VOLUME = 10 * 10000               # 10万股


def _market_from_code(code: str) -> str:
    """根据股票代码判断市场"""
    if code[0] in ('6', '9'):
        return 'SH'
    else:
        return 'SZ'


def _build_wind_code(code: str) -> str:
    """构建万得代码: 000001 -> 000001.SZ"""
    mkt = _market_from_code(code)
    return f"{code}.{mkt}"


def read_level2_order(file_path: str | Path) -> pd.DataFrame:
    """读取逐笔委托CSV（GB18030编码）"""
    fp = Path(file_path)
    if not fp.exists():
        logger.error(f"文件不存在: {fp}")
        return pd.DataFrame()

    df = pd.read_csv(
        fp, encoding='gb18030', dtype=WTDATA_TYPES, low_memory=False,
    )
    return df


def read_level2_trade(file_path: str | Path) -> pd.DataFrame:
    """读取逐笔成交CSV（GB18030编码），过滤撤单，添加成交金额"""
    fp = Path(file_path)
    if not fp.exists():
        logger.error(f"文件不存在: {fp}")
        return pd.DataFrame()

    df = pd.read_csv(
        fp, encoding='gb18030', dtype=CJDATA_TYPES, low_memory=False,
    )
    # 深圳成交代码=C是撤单，上海无撤单代码
    # 成交价格<=0的记录无效
    df = df[df['成交价格'] > 0].copy()
    if '成交代码' in df.columns:
        df = df[df['成交代码'] != 'C'].copy()

    df['成交金额'] = df['成交价格'] * df['成交数量']
    return df


def read_level2_quote(file_path: str | Path) -> pd.DataFrame:
    """读取行情快照CSV（GB18030编码）"""
    fp = Path(file_path)
    if not fp.exists():
        logger.error(f"文件不存在: {fp}")
        return pd.DataFrame()

    df = pd.read_csv(
        fp, encoding='gb18030', dtype=HQDATA_TYPES, low_memory=False,
    )
    return df


def read_level2_stock_dir(stock_dir: str | Path) -> dict[str, pd.DataFrame]:
    """读取某只股票一个交易日的全部Level-2数据"""
    sd = Path(stock_dir)
    result = {}
    readers = {
        '逐笔委托': (sd / '逐笔委托.csv', read_level2_order),
        '逐笔成交': (sd / '逐笔成交.csv', read_level2_trade),
        '行情': (sd / '行情.csv', read_level2_quote),
    }
    for key, (fp, reader) in readers.items():
        if fp.exists():
            result[key] = reader(fp)
    return result


def read_level2_day(
    data_root: str | Path,
    date: str,
    code: str,
) -> dict[str, pd.DataFrame]:
    """
    读取指定日期+代码的Level-2数据

    data_root: Level-2数据根目录 (e.g. 'e:/' 或 '/Volumes/data/')
    date: '2026-01-05'
    code: '000001'

    期望目录结构: {data_root}/{YYYYMMDD}/{code.MARKET}/*.csv
    """
    day = date.replace('-', '')
    wind_code = _build_wind_code(code)
    stock_dir = Path(data_root) / day / wind_code
    return read_level2_stock_dir(stock_dir)


# ═══════════════════════════════════════════════════════════════
# 委托-成交匹配
# ═══════════════════════════════════════════════════════════════

def _usable_id_ratio(series: pd.Series) -> float:
    """Return ratio of non-empty, non-zero order ids."""
    if series.empty:
        return 0.0
    values = series.astype(str).str.strip()
    usable = values.notna() & (values != "") & (values != "0") & (values.str.lower() != "nan")
    return float(usable.mean())


def _choose_order_match_key(wtdf: pd.DataFrame) -> str:
    """
    Choose the order id column used by trade buy/sell sequence numbers.

    Older samples use 委托编号. The 2025 Wind archives have 委托编号=0 and
    require 交易所委托号 instead.
    """
    candidates = ["委托编号", "交易所委托号"]
    ratios = {col: _usable_id_ratio(wtdf[col]) for col in candidates if col in wtdf.columns}
    if not ratios:
        raise KeyError("逐笔委托缺少 委托编号/交易所委托号")
    return max(ratios, key=ratios.get)


def match_orders_to_trades(wtdf: pd.DataFrame, cjdf: pd.DataFrame) -> pd.DataFrame:
    """
    将逐笔委托与逐笔成交匹配，得到成交的委托明细。

    默认用 委托编号；当 委托编号 基本不可用（如2025样本全为0）时，
    自动回退到 交易所委托号。

    wtdf: 逐笔委托 DataFrame
    cjdf: 逐笔成交 DataFrame（已过滤撤单）
    Returns: 匹配成功的委托 DataFrame，含成交数量/金额列
    """
    if wtdf.empty or cjdf.empty:
        return pd.DataFrame()

    match_key = _choose_order_match_key(wtdf)

    buy_agg = cjdf.groupby('叫买序号', dropna=True).agg({'成交数量': 'sum', '成交金额': 'sum'})
    sell_agg = cjdf.groupby('叫卖序号', dropna=True).agg({'成交数量': 'sum', '成交金额': 'sum'})
    cj_agg = pd.concat([buy_agg, sell_agg])
    cj_agg = cj_agg.groupby(cj_agg.index).agg({'成交数量': 'sum', '成交金额': 'sum'})
    cj_agg.insert(0, match_key, cj_agg.index.astype(str))

    # 去除沪市撤单类型
    wt_valid = wtdf[wtdf['委托类型'] != 'D'].copy() if '委托类型' in wtdf.columns else wtdf.copy()

    wt_valid[match_key] = wt_valid[match_key].astype(str)
    matched = pd.merge(wt_valid, cj_agg, on=match_key, how='inner')
    matched['match_key'] = match_key
    matched['委托金额'] = matched['委托数量'] * matched['委托价格']
    return matched


# ═══════════════════════════════════════════════════════════════
# 大单/超大单分类
# ═══════════════════════════════════════════════════════════════

def classify_orders_by_size(
    wtcj: pd.DataFrame,
    super_amount: int = SUPER_AMOUNT,
    super_volume: float = SUPER_VOLUME,
    big_amount: int = BIG_AMOUNT,
    big_volume: float = BIG_VOLUME,
) -> dict[str, pd.DataFrame]:
    """
    按金额/数量将成交委托分为超大单、大单、小单

    wtcj: match_orders_to_trades() 的输出

    Returns: {'super': df, 'big': df, 'small': df}
    """
    if wtcj.empty:
        return {'super': pd.DataFrame(), 'big': pd.DataFrame(), 'small': pd.DataFrame()}

    wtcj = wtcj.copy()
    if '委托金额' not in wtcj.columns and {'委托价格', '委托数量'}.issubset(wtcj.columns):
        wtcj['委托金额'] = wtcj['委托价格'].astype(float) * wtcj['委托数量'].astype(float)

    is_super = (wtcj['委托金额'] >= super_amount) | (wtcj['委托数量'] >= super_volume)
    is_big = (wtcj['委托金额'] >= big_amount) | (wtcj['委托数量'] >= big_volume)

    return {
        'super': wtcj[is_super].copy(),
        'big': wtcj[is_big & ~is_super].copy(),
        'small': wtcj[~is_big].copy(),
    }


def compute_big_order_summary(wtcj: pd.DataFrame) -> dict:
    """
    计算大单/超大单买卖汇总

    Returns: dict with super_buy, super_sell, big_buy, big_sell (单位: 亿元)
             and mean prices
    """
    if wtcj.empty:
        return {
            'super_buy': 0, 'super_sell': 0,
            'big_buy': 0, 'big_sell': 0,
            'super_buy_avg_price': 0, 'super_sell_avg_price': 0,
            'big_buy_avg_price': 0, 'big_sell_avg_price': 0,
        }

    classified = classify_orders_by_size(wtcj)

    def _buy_sell_amount_volume(df):
        if df.empty:
            return 0, 0, 0, 0
        buy = df[df['委托代码'] == 'B']
        sell = df[df['委托代码'] == 'S']
        b_amt_raw = buy['成交金额'].sum()
        s_amt_raw = sell['成交金额'].sum()
        b_vol = buy['成交数量'].sum()
        s_vol = sell['成交数量'].sum()
        return b_amt_raw, s_amt_raw, b_vol, s_vol

    def _amt_yi(raw_amount):
        # 成交价格=元×10000，成交金额=成交价格×股数，所以 /1e12=亿元。
        return round(raw_amount / 1e12, 2)

    def _avg_price(raw_amount, vol):
        if vol > 0:
            return round(raw_amount / vol / 10000, 2)  # 转元
        return 0

    sb, ss, sbv, ssv = _buy_sell_amount_volume(classified['super'])
    bb, bs, bbv, bsv = _buy_sell_amount_volume(classified['big'])

    return {
        'super_buy': _amt_yi(sb), 'super_sell': _amt_yi(ss),
        'big_buy': _amt_yi(bb), 'big_sell': _amt_yi(bs),
        'super_buy_avg_price': _avg_price(sb, sbv),
        'super_sell_avg_price': _avg_price(ss, ssv),
        'big_buy_avg_price': _avg_price(bb, bbv),
        'big_sell_avg_price': _avg_price(bs, bsv),
    }


# ═══════════════════════════════════════════════════════════════
# 日线汇总（从逐笔成交推算OHLCV）
# ═══════════════════════════════════════════════════════════════

def compute_daily_ohlcv(cjdf: pd.DataFrame) -> dict:
    """
    从逐笔成交计算日线OHLCV（比行情快照更准确，已剔除撤单）

    Returns: {close, open, high, low, amount(亿), volume(万股)}
      成交价格=元×10000, 成交金额=成交价格×成交数量 → /1e12=亿元
    """
    if cjdf.empty:
        return {'close': 0, 'open': 0, 'high': 0, 'low': 0, 'amount': 0, 'volume': 0}

    prices = cjdf['成交价格'].values
    return {
        'close': prices[-1] / 10000,
        'open': prices[0] / 10000,
        'high': prices.max() / 10000,
        'low': prices.min() / 10000,
        'amount': round(cjdf['成交金额'].sum() / 1e12, 2),   # 亿 (raw price是×10000)
        'volume': cjdf['成交数量'].sum() / 10000,             # 万股
    }


def compute_minute_bars(cjdf: pd.DataFrame) -> pd.DataFrame:
    """从逐笔成交聚合为分钟K线"""
    if cjdf.empty:
        return pd.DataFrame()

    cj = cjdf.copy()
    cj['分钟'] = cj['时间'].str[:4]  # 前4位 = HHMM
    cj['价格_元'] = cj['成交价格'] / 10000

    bars = cj.groupby('分钟').agg(
        开盘=('价格_元', 'first'),
        收盘=('价格_元', 'last'),
        最高=('价格_元', 'max'),
        最低=('价格_元', 'min'),
        成交额_万=('成交金额', lambda x: x.sum() / 1e8),
        成交量_手=('成交数量', lambda x: x.sum() / 100),
    ).reset_index()
    return bars


# ═══════════════════════════════════════════════════════════════
# 资金流向分析
# ═══════════════════════════════════════════════════════════════

def compute_period_flow(
    cjdf: pd.DataFrame,
    start_time: str = '0925',
    end_time: str = '1500',
) -> dict:
    """
    计算某时段的资金流向

    start_time/end_time: '0925', '1130', '1500' 等
    """
    cj = cjdf.copy()
    cj['时间_int'] = cj['时间'].astype(int)
    start = int(start_time) * 100000
    end = int(end_time) * 100000
    period = cj[(cj['时间_int'] >= start) & (cj['时间_int'] <= end)]

    if period.empty:
        return {
            'start_time': start_time, 'end_time': end_time,
            'volume_wan': 0, 'amount_yi': 0,
            'buy_delegates': 0, 'sell_delegates': 0,
            'super_buy': 0, 'super_sell': 0,
            'big_buy': 0, 'big_sell': 0,
            'open': 0, 'close': 0, 'high': 0, 'low': 0,
        }

    prices = period['成交价格'].values
    volume_wan = period['成交数量'].sum() / 10000
    amount_yi = period['成交金额'].sum() / 1e12

    buy_grp = period.groupby('叫买序号')
    sell_grp = period.groupby('叫卖序号')
    buy_amts = buy_grp['成交金额'].sum()
    sell_amts = sell_grp['成交金额'].sum()

    super_buy = buy_amts[buy_amts > SUPER_AMOUNT].sum() / 1e12
    super_sell = sell_amts[sell_amts > SUPER_AMOUNT].sum() / 1e12
    big_buy = buy_amts[buy_amts > BIG_AMOUNT].sum() / 1e12
    big_sell = sell_amts[sell_amts > BIG_AMOUNT].sum() / 1e12

    return {
        'start_time': start_time, 'end_time': end_time,
        'volume_wan': round(volume_wan, 2),
        'amount_yi': round(amount_yi, 2),
        'buy_delegates': len(buy_grp),
        'sell_delegates': len(sell_grp),
        'super_buy': round(super_buy, 2),
        'super_sell': round(super_sell, 2),
        'big_buy': round(big_buy, 2),
        'big_sell': round(big_sell, 2),
        'open': prices[0] / 10000,
        'close': prices[-1] / 10000,
        'high': prices.max() / 10000,
        'low': prices.min() / 10000,
    }


def filter_buy_orders(wtdf: pd.DataFrame) -> pd.DataFrame:
    """筛选买入委托"""
    if wtdf.empty or '委托代码' not in wtdf.columns:
        return wtdf
    return wtdf[wtdf['委托代码'] == 'B'].copy()


def filter_buy_trades(cjdf: pd.DataFrame) -> pd.DataFrame:
    """筛选主动买入成交（BS标志=B）"""
    if cjdf.empty or 'BS标志' not in cjdf.columns:
        return cjdf
    return cjdf[cjdf['BS标志'] == 'B'].copy()


def filter_sell_trades(cjdf: pd.DataFrame) -> pd.DataFrame:
    """筛选主动卖出成交（BS标志=S）"""
    if cjdf.empty or 'BS标志' not in cjdf.columns:
        return cjdf
    return cjdf[cjdf['BS标志'] == 'S'].copy()
