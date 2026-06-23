"""
匿名机构追踪器
基于拆单聚类结果，跨交易日追踪匿名机构，计算每个机构的Alpha表现。

核心流程:
  同日: 聚类 → "机构A在股票X以价格P买入金额M"（split_detector）
  跨日: 行为指纹匹配 → 同一机构在不同日期的操作归拢
  Alpha: 计算机构买入后N日收益，累积跟踪记录
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 跨日匹配相似度阈值（余弦相似度 > 此值视为同一机构）
FINGERPRINT_MATCH_THRESHOLD = 0.85


def extract_fingerprint(op: dict) -> np.ndarray:
    """
    从单次机构操作提取行为指纹向量

    特征（6维，已归一化）:
      [0] log_median_order_size: 单笔委托量中位数(log10)
      [1] order_size_cv: 委托量变异系数(std/mean)，衡量拆单均匀度
      [2] time_preference: 交易时段偏好 [-1,1]，负=早盘，正=尾盘
      [3] price_aggressiveness: VWAP偏离度 [-1,1]，正=追涨，负=低吸
      [4] log_slice_count: 拆单数量(log10)
      [5] log_time_span: 操作时间跨度(log10分钟)
    """
    import math
    ops = op.get('_order_details', None)
    if ops is not None and not ops.empty:
        qtys = ops['委托数量'].astype(float).values
        times = ops['时间'].astype(str).apply(_time_to_seconds_float).values
        prices = ops['委托价格'].values / 10000
        med_qty = np.median(qtys)
        cv = np.std(qtys) / np.mean(qtys) if np.mean(qtys) > 0 else 0
        trading_mid = 12 * 3600
        time_dev = (np.mean(times) - trading_mid) / (3 * 3600)
        price_agg = op.get('vwap_deviation_pct', 0) / 200
        return np.array([
            np.log10(max(med_qty, 1)),
            np.clip(cv, 0, 3),
            np.clip(time_dev, -1, 1),
            np.clip(price_agg, -1, 1),
            np.log10(max(len(qtys), 1)),
            np.log10(max((times.max() - times.min()) / 60, 0.1)),
        ])

    # 从汇总统计近似估算指纹
    avg_size = op.get('avg_order_size_wan', 5000)  # 万元
    time_hour = (op.get('start_time', 34200) + op.get('end_time', 54000)) / 2
    time_dev = (time_hour - 12 * 3600) / (3 * 3600)
    return np.array([
        np.log10(max(avg_size * 10000, 1)),          # 转回股数的log10
        np.clip(op.get('time_span_min', 30) / 60, 0, 3),  # 跨度作CV近似
        np.clip(time_dev, -1, 1),
        np.clip(op.get('vwap_deviation_pct', 0) / 200, -1, 1),
        np.log10(max(op.get('order_count', 3), 1)),
        np.log10(max(op.get('time_span_min', 1), 0.1)),
    ])

    med_qty = np.median(qtys)
    cv = np.std(qtys) / np.mean(qtys) if np.mean(qtys) > 0 else 0
    trading_mid = 12 * 3600
    time_dev = (np.mean(times) - trading_mid) / (3 * 3600)  # ±3小时
    price_agg = op.get('vwap_deviation_pct', 0) / 200  # ±2%范围

    features = np.array([
        np.log10(max(med_qty, 1)),
        np.clip(cv, 0, 3),
        np.clip(time_dev, -1, 1),
        np.clip(price_agg, -1, 1),
        np.log10(max(len(qtys), 1)),
        np.log10(max((times.max() - times.min()) / 60, 0.1)),
    ])
    return features


def _time_to_seconds_float(time_str: str) -> float:
    t = str(time_str).zfill(9)
    h, m, s = int(t[0:2]), int(t[2:4]), int(t[4:6])
    ms = int(t[6:9]) / 1000
    return h * 3600 + m * 60 + s + ms


def match_institution(
    fingerprint: np.ndarray,
    institution_db: dict[str, np.ndarray],
    threshold: float = FINGERPRINT_MATCH_THRESHOLD,
) -> str | None:
    """
    将行为指纹与已有机构库匹配，返回最相似的机构ID（或None=新机构）

    institution_db: {inst_id: fingerprint_vector}
    """
    if not institution_db or len(fingerprint) == 0:
        return None

    best_id, best_sim = None, -1
    for inst_id, db_fp in institution_db.items():
        sim = _cosine_similarity(fingerprint, db_fp)
        if sim > best_sim:
            best_sim, best_id = sim, inst_id

    if best_sim >= threshold:
        return best_id
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class InstitutionTracker:
    """
    匿名机构追踪数据库

    跨日累积：同一行为指纹的机构操作归拢，追踪Alpha表现
    """

    def __init__(self, match_threshold: float = FINGERPRINT_MATCH_THRESHOLD):
        self._next_id = 0
        self.match_threshold = match_threshold
        self.fingerprints: dict[str, np.ndarray] = {}   # inst_id → 行为指纹（滚动平均）
        self.records: dict[str, list[dict]] = defaultdict(list)  # inst_id → [操作记录]
        self.alpha: dict[str, dict] = {}                 # inst_id → Alpha统计

    def register_operation(
        self,
        op: dict,
        date: str,
        stock_code: str,
    ) -> str:
        """
        注册一次机构操作，自动匹配或创建机构ID
        """
        fp = extract_fingerprint(op)
        inst_id = match_institution(fp, self.fingerprints, threshold=self.match_threshold)

        if inst_id is None:
            inst_id = f"INST_{self._next_id:04d}"
            self._next_id += 1
            self.fingerprints[inst_id] = fp
        else:
            # 滚动平均更新指纹
            alpha = 0.3
            self.fingerprints[inst_id] = (
                alpha * fp + (1 - alpha) * self.fingerprints[inst_id]
            )

        record = {
            **op,
            'date': date,
            'stock_code': stock_code,
        }
        self.records[inst_id].append(record)
        return inst_id

    def calculate_forward_returns(
        self,
        price_loader,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        为所有机构操作计算买入后N日收益率

        price_loader: callable(code, start_date, end_date) -> DataFrame[date, close, open]
        horizons: [1, 5, 10, 20]
        """
        horizons = horizons or [1, 5, 10, 20]
        all_returns = []

        for inst_id, ops in self.records.items():
            for op in ops:
                stock = op['stock_code']
                date = op['date']
                avg_price = op.get('avg_price', 0)
                if avg_price <= 0:
                    continue

                try:
                    prices = price_loader(
                        stock,
                        start_date=date,
                        end_date=(pd.to_datetime(date) + pd.Timedelta(days=max(horizons) + 5)).strftime('%Y-%m-%d'),
                    )
                except Exception:
                    continue

                if prices.empty:
                    continue

                prices = prices[prices['date'] >= date].reset_index(drop=True)
                if len(prices) < 2:
                    continue

                entry_price = prices.iloc[0]['open']

                row = {
                    'institution_id': inst_id,
                    'stock_code': stock,
                    'entry_date': date,
                    'avg_price': avg_price,
                    'entry_open': entry_price,
                    'total_amount_wan': op.get('total_amount_wan', 0),
                }
                for h in horizons:
                    if h < len(prices):
                        row[f'ret_{h}d'] = round(
                            (prices.iloc[h]['close'] - entry_price) / entry_price, 4
                        )
                    else:
                        row[f'ret_{h}d'] = np.nan

                all_returns.append(row)

        return pd.DataFrame(all_returns)

    def summarize_alpha(self, returns_df: pd.DataFrame) -> dict:
        """统计每个机构的Alpha表现"""
        if returns_df.empty:
            return {}

        alpha_summary = {}
        for inst_id, group in returns_df.groupby('institution_id'):
            rets_20d = group['ret_20d'].dropna()
            rets_10d = group['ret_10d'].dropna()
            rets_5d = group['ret_5d'].dropna()

            alpha_summary[inst_id] = {
                'total_operations': len(group),
                'total_amount_wan': round(group['total_amount_wan'].sum(), 2),
                'avg_ret_5d': round(rets_5d.mean(), 4) if len(rets_5d) > 0 else 0,
                'avg_ret_10d': round(rets_10d.mean(), 4) if len(rets_10d) > 0 else 0,
                'avg_ret_20d': round(rets_20d.mean(), 4) if len(rets_20d) > 0 else 0,
                'win_rate_20d': round((rets_20d > 0).mean(), 4) if len(rets_20d) > 0 else 0,
                'sharpe_20d': round(
                    rets_20d.mean() / rets_20d.std() * np.sqrt(252 / 20), 4
                ) if len(rets_20d) > 2 and rets_20d.std() > 0 else 0,
            }

        return alpha_summary

    def top_institutions(self, alpha_summary: dict, n: int = 20, metric: str = 'avg_ret_20d') -> list:
        """返回Alpha最高的N个机构"""
        sorted_insts = sorted(
            alpha_summary.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True,
        )
        return [(inst_id, stats) for inst_id, stats in sorted_insts[:n]]

    def get_records_df(self) -> pd.DataFrame:
        """所有机构操作记录合并为DataFrame"""
        all_records = []
        for inst_id, ops in self.records.items():
            for op in ops:
                op['institution_id'] = inst_id
                all_records.append(op)
        return pd.DataFrame(all_records)

    def stats(self) -> dict:
        return {
            'total_institutions': len(self.fingerprints),
            'total_operations': sum(len(ops) for ops in self.records.values()),
        }
