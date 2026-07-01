"""
拆单检测器
基于DBSCAN无监督聚类，对逐笔委托-成交匹配后的大单买入进行聚类，
检测匿名机构的拆单行为。

方法论:
  - ALLO: 先过滤大单(排除散户噪声) → 再基于时间窗口+数量模式+价格区间检测
  - ClusterLOB: 时间/价格/数量三维特征 + DBSCAN + OFI验证

流程:
  1. 过滤: 仅保留 委托金额>=大单阈值(20万元) 或 委托数量>=大单阈值(10万股) 的买单
  2. 聚类: 时间+价格偏离+数量(log10) 三维DBSCAN
  3. 汇总: 每cluster = 一个匿名机构的一次买入操作
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 交易时段
OPEN_TIME = 9 * 3600 + 30 * 60       # 09:30
CLOSE_TIME = 15 * 3600               # 15:00
TRADING_SECONDS = CLOSE_TIME - OPEN_TIME  # 19800

# 大单过滤阈值（金额单位: 分, 数量单位: 股）
BIG_AMOUNT = 20 * 10000 * 10000       # 20万元
BIG_VOLUME = 10 * 10000               # 10万股


def _time_to_seconds(time_str: str) -> float:
    t = str(time_str).zfill(9)
    h, m, s = int(t[0:2]), int(t[2:4]), int(t[4:6])
    ms = int(t[6:9]) / 1000
    return h * 3600 + m * 60 + s + ms


def detect_institution_operations(
    wtcj: pd.DataFrame,
    eps: float = 0.15,
    min_samples: int = 5,
    min_total_amount_wan: float = 100,
    big_amount: int = BIG_AMOUNT,
    big_volume: float = BIG_VOLUME,
) -> list[dict]:
    """
    对单只股票单日的成交委托进行DBSCAN聚类，检测机构拆单操作

    wtcj: match_orders_to_trades() 输出，含委托代码/委托价格/委托数量/成交金额等
    eps: DBSCAN邻域半径（RobustScaler后），越小越严格。0.1=紧, 0.3=松
    min_samples: 最少拆单笔数，低于此数视为散户
    min_total_amount_wan: 聚类最少总成交金额（万元）
    big_amount: 大单金额阈值（分），默认20万
    big_volume: 大单数量阈值（股），默认10万

    Returns: [{cluster_id, direction, total_amount_wan, avg_price,
               order_count, time_span_min, ...}] 按总金额降序
    """
    if wtcj.empty:
        return []

    df = wtcj.copy()

    # ---- Step 1: 过滤零价 + 大单 ----
    df = df[df['委托价格'] > 0]
    if df.empty:
        return []

    df['委托金额'] = df['委托价格'].astype(float) * df['委托数量'].astype(float)
    is_big = (df['委托金额'] >= big_amount) | (df['委托数量'] >= big_volume)
    df_big = df[is_big].copy()
    if df_big.empty:
        return []

    # 分买卖方向
    results = []
    for direction, direction_label in [('B', 'BUY'), ('S', 'SELL')]:
        df_dir = df_big[df_big['委托代码'] == direction]
        if len(df_dir) < min_samples:
            continue

        ops = _cluster_direction(
            df_dir, direction_label, eps, min_samples,
            min_total_amount_wan,
        )
        results.extend(ops)

    results.sort(key=lambda x: x['total_amount_wan'], reverse=True)
    return results


def _cluster_direction(
    df: pd.DataFrame,
    direction: str,
    eps: float,
    min_samples: int,
    min_total_amount_wan: float,
) -> list[dict]:
    """对单方向（买或卖）的大单聚类"""

    # ---- 特征 ----
    time_sec = df['时间'].astype(str).apply(_time_to_seconds)
    time_norm = ((time_sec - OPEN_TIME) / TRADING_SECONDS).values.reshape(-1, 1)

    # 从成交数据计算VWAP
    if '成交价格' in df.columns and '成交数量' in df.columns:
        vwap = (df['成交价格'] * df['成交数量']).sum() / df['成交数量'].sum()
    else:
        vwap = df['委托价格'].mean()
    vwap_yuan = vwap / 10000

    price_yuan = df['委托价格'].values / 10000
    price_dev = ((price_yuan - vwap_yuan) / vwap_yuan).reshape(-1, 1)

    qty = df['委托数量'].values.astype(float)
    log_qty = np.log10(np.clip(qty, 100, None)).reshape(-1, 1)

    # ---- RobustScaler（抗离群值） + DBSCAN ----
    features = np.hstack([time_norm, price_dev, log_qty])
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(features_scaled)

    # ---- 汇总 ----
    results = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        cluster = df[labels == cid]
        n = len(cluster)
        if n < min_samples:
            continue

        times = cluster['时间'].astype(str).apply(_time_to_seconds)
        prices = cluster['委托价格'].values / 10000
        volumes = cluster['委托数量'].astype(float).values
        weights = prices * volumes

        # 成交金额=成交价格(×10000)*成交数量 → /1e8=万元
        trade_amt_raw = cluster['成交金额'].sum() if '成交金额' in cluster.columns else weights.sum()
        total_trade_amt_wan = trade_amt_raw / 1e8

        if total_trade_amt_wan < min_total_amount_wan:
            continue

        avg_price = weights.sum() / volumes.sum() if volumes.sum() > 0 else 0
        time_span = (times.max() - times.min()) / 60

        results.append({
            'cluster_id': int(cid),
            'direction': direction,
            'total_amount_wan': round(total_trade_amt_wan, 2),
            'avg_price': round(float(avg_price), 2),
            'order_count': n,
            'time_span_min': round(time_span, 1),
            'start_time': int(times.min()),
            'end_time': int(times.max()),
            'buy_volume_wan': round(volumes.sum() / 10000, 2),
            'price_min': round(float(prices.min()), 2),
            'price_max': round(float(prices.max()), 2),
            'vwap_deviation_pct': round(float((avg_price - vwap_yuan) / vwap_yuan * 100), 2),
            'avg_order_size_wan': round(weights.sum() / n / 10000, 2),
            'median_order_qty': round(float(np.median(volumes)), 0),
            'qty_cv': round(float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0, 2),
            'mid_time_sec': int((times.min() + times.max()) / 2),
            # 订单时间间隔波动（秒），衡量拆单节奏是机械等间隔还是随机
            'order_interval_std': round(float(np.std(np.diff(np.sort(times.values)))) if n > 1 else 0, 1),
            # HHI拆单集中度：1=全部集中一笔，≈1/n=完全均匀拆分
            'order_hhi': round(float(np.sum((volumes / volumes.sum()) ** 2)), 4),
        })

    return results


def detect_all_stocks(
    day_data: dict[str, dict[str, pd.DataFrame]],
    eps: float = 0.15,
    min_samples: int = 5,
    min_total_amount_wan: float = 100,
) -> pd.DataFrame:
    """
    对全部股票的Level-2数据进行拆单检测

    day_data: {stock_code: {'逐笔委托': df, '逐笔成交': df, '行情': df}, ...}
    Returns: DataFrame of all detected institution operations
    """
    from ..data.level2_reader import match_orders_to_trades

    all_ops = []
    for code, data in day_data.items():
        if '逐笔委托' not in data or '逐笔成交' not in data:
            continue
        wtcj = match_orders_to_trades(data['逐笔委托'], data['逐笔成交'])
        if wtcj.empty:
            continue
        ops = detect_institution_operations(
            wtcj, eps=eps, min_samples=min_samples,
            min_total_amount_wan=min_total_amount_wan,
        )
        for op in ops:
            op['stock_code'] = code
        all_ops.extend(ops)

    df = pd.DataFrame(all_ops)
    if not df.empty:
        df = df.sort_values('total_amount_wan', ascending=False).reset_index(drop=True)
    return df

