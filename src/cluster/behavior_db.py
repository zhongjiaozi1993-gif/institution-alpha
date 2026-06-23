"""
行为模式数据库（B方案核心）

不关心"谁"在买，只关心"怎么买"：
  检测拆单聚类 → 提取行为指纹 → N天后回看收益 → 积累行为-收益对
  → 新聚类来了 → 找历史相似行为 → 预测Alpha → 跟买信号

与 institution_tracker 的区别:
  - institution_tracker 试图跨日追踪同一机构身份（需要席位号）
  - behavior_db 只管行为模式是否相似，不追踪身份
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 行为指纹维度说明:
#   [0] log_median_qty: 中位数委托数量(log10股)
#   [1] qty_cv: 委托数量变异系数(std/mean)，衡量拆单均匀度
#   [2] time_preference: 交易时段偏好[-1,1]，负=早盘，正=尾盘
#   [3] price_aggressiveness: VWAP偏离度[-1,1]，正=追涨，负=低吸
#   [4] log_slice_count: 拆单数量(log10)
#   [5] log_time_span: 操作时间跨度(log10分钟)


def extract_behavior_fp(op: dict) -> np.ndarray:
    """
    从一次聚类操作提取标准化行为指纹（6维，均归一化到合理范围）

    各维度已做clip处理，适合直接用于余弦相似度计算
    """
    # 委托数量特征
    med_qty = op.get('median_order_qty', op.get('total_amount_wan', 100) * 10000 / (op.get('avg_price', 10) or 10))
    if med_qty <= 0:
        med_qty = 10000
    cv = op.get('qty_cv', 0.5)

    # 时间特征
    mid_time = op.get('mid_time_sec', 0)
    if mid_time <= 0:
        mid_time = (op.get('start_time', 34200) + op.get('end_time', 54000)) / 2
    time_dev = (mid_time - 12 * 3600) / (3 * 3600)

    # 价格特征
    price_agg = (op.get('vwap_deviation_pct', 0) or 0) / 200

    # 规模特征
    n_slices = max(op.get('order_count', 3), 1)
    time_span = max(op.get('time_span_min', 1), 0.1)

    return np.array([
        np.log10(max(med_qty, 100)),
        np.clip(cv, 0, 3),
        np.clip(time_dev, -1, 1),
        np.clip(price_agg, -1, 1),
        np.log10(n_slices),
        np.log10(max(time_span, 0.1)),
    ], dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BehaviorDB:
    """
    行为模式数据库

    存储: [fingerprint(6维), 操作特征, 收益率结果, 元数据]
    查询: 找K个最相似的历史模式 → 相似度加权预测Alpha
    """

    def __init__(self):
        self.fingerprints: list[np.ndarray] = []          # N × 6
        self.metadata: list[dict] = []                    # 操作特征
        self.returns: list[dict[str, float]] = []          # {ret_5d, ret_10d, ret_20d}
        self._fp_matrix: np.ndarray | None = None          # 缓存

    def add(self, fp: np.ndarray, returns: dict[str, float], meta: dict):
        self.fingerprints.append(fp.astype(np.float64))
        self.returns.append(returns)
        self.metadata.append(meta)
        self._fp_matrix = None  # 失效缓存

    def add_batch(self, fps: list[np.ndarray], returns_list: list[dict], metas: list[dict]):
        for fp, rets, meta in zip(fps, returns_list, metas):
            self.add(fp, rets, meta)

    def _build_matrix(self):
        if self._fp_matrix is None and len(self.fingerprints) > 0:
            self._fp_matrix = np.vstack(self.fingerprints)

    def __len__(self):
        return len(self.fingerprints)

    def find_similar(
        self,
        query_fp: np.ndarray,
        top_k: int = 20,
        min_similarity: float = 0.7,
        exclude_indices: set[int] | None = None,
    ) -> list[dict]:
        """
        查找最相似的K个历史行为模式

        Returns: [{idx, similarity, returns, meta}, ...] 按相似度降序
        """
        if len(self.fingerprints) == 0:
            return []

        self._build_matrix()
        exclude = exclude_indices or set()

        # 计算余弦相似度
        fp_norm = query_fp / (np.linalg.norm(query_fp) or 1)
        db_norm = self._fp_matrix / (np.linalg.norm(self._fp_matrix, axis=1, keepdims=True) + 1e-10)
        sims = np.dot(db_norm, fp_norm)

        # 排序取top_k
        order = np.argsort(sims)[::-1]
        results = []
        for idx in order:
            if idx in exclude:
                continue
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            results.append({
                'idx': int(idx),
                'similarity': round(sim, 4),
                'returns': self.returns[idx],
                'meta': self.metadata[idx],
            })
            if len(results) >= top_k:
                break

        return results

    def predict(
        self,
        query_fp: np.ndarray,
        horizon: str = 'ret_20d',
        top_k: int = 20,
        min_similarity: float = 0.7,
        min_samples: int = 5,
        exclude_indices: set[int] | None = None,
    ) -> dict | None:
        """
        基于相似历史行为模式预测Alpha

        Returns: {
            predicted_return, confidence, n_samples, mean_similarity,
            win_rate, sharpe_like, prediction_strength
        } or None if insufficient similar samples
        """
        similar = self.find_similar(
            query_fp, top_k=top_k, min_similarity=min_similarity,
            exclude_indices=exclude_indices,
        )

        if len(similar) < min_samples:
            return None

        horizon_returns = np.array([s['returns'].get(horizon, np.nan) for s in similar])
        valid = ~np.isnan(horizon_returns)
        if valid.sum() < min_samples:
            return None

        rets = horizon_returns[valid]
        sims = np.array([s['similarity'] for s in similar])[valid]

        # 相似度加权平均
        weights = sims / sims.sum()
        weighted_ret = float(np.dot(rets, weights))
        weighted_win_rate = float(np.dot((rets > 0).astype(float), weights))

        # Sharpe-like: weighted_ret / weighted_std
        weighted_std = np.sqrt(np.dot(weights, (rets - weighted_ret) ** 2))
        sharpe_like = weighted_ret / weighted_std if weighted_std > 0 else 0.0

        return {
            'predicted_return': round(weighted_ret, 4),
            'win_rate': round(weighted_win_rate, 4),
            'n_samples': int(valid.sum()),
            'mean_similarity': round(float(sims.mean()), 4),
            'sharpe_like': round(sharpe_like, 4),
            'prediction_strength': round(
                weighted_ret * weighted_win_rate * min(valid.sum() / 20, 1.0), 4
            ),
        }

    def cross_validate(
        self,
        horizon: str = 'ret_20d',
        top_k: int = 20,
        min_similarity: float = 0.7,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        """
        留一法交叉验证：对DB中每条记录，用其他记录预测它，对比预测vs实际

        Returns: DataFrame with actual, predicted, error columns
        """
        results = []
        for i in range(len(self.fingerprints)):
            pred = self.predict(
                self.fingerprints[i], horizon=horizon,
                top_k=top_k, min_similarity=min_similarity,
                min_samples=min_samples, exclude_indices={i},
            )
            if pred is None:
                continue

            actual = self.returns[i].get(horizon, np.nan)
            if np.isnan(actual):
                continue

            results.append({
                **self.metadata[i],
                'predicted': pred['predicted_return'],
                'actual': actual,
                'error': pred['predicted_return'] - actual,
                'n_similar': pred['n_samples'],
                'mean_similarity': pred['mean_similarity'],
                'pred_strength': pred['prediction_strength'],
            })

        return pd.DataFrame(results)

    def stats(self) -> dict:
        """基本统计"""
        if len(self) == 0:
            return {'total_patterns': 0}

        horizons = ['ret_5d', 'ret_10d', 'ret_20d']
        ret_stats = {}
        for h in horizons:
            vals = [r.get(h, np.nan) for r in self.returns]
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                ret_stats[h] = {
                    'mean': round(float(np.mean(valid)), 4),
                    'median': round(float(np.median(valid)), 4),
                    'std': round(float(np.std(valid)), 4),
                    'win_rate': round(float((np.array(valid) > 0).mean()), 4),
                    'n': len(valid),
                }

        return {
            'total_patterns': len(self),
            'returns': ret_stats,
        }

    def save(self, path: str | Path):
        """保存到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fps_arr = np.vstack(self.fingerprints) if self.fingerprints else np.array([])
        rets_df = pd.DataFrame(self.returns)
        meta_df = pd.DataFrame(self.metadata)

        np.savez_compressed(path.with_suffix('.npz'), fingerprints=fps_arr)
        rets_df.to_parquet(path.with_name(path.stem + '_returns.parquet'), index=False)
        meta_df.to_parquet(path.with_name(path.stem + '_meta.parquet'), index=False)

    @classmethod
    def load(cls, path: str | Path) -> 'BehaviorDB':
        """从文件加载"""
        path = Path(path)
        db = cls()

        npz = np.load(path.with_suffix('.npz'))
        fps = npz['fingerprints']
        rets_df = pd.read_parquet(path.with_name(path.stem + '_returns.parquet'))
        meta_df = pd.read_parquet(path.with_name(path.stem + '_meta.parquet'))

        db.fingerprints = [fps[i] for i in range(len(fps))]
        db.returns = rets_df.to_dict('records')
        db.metadata = meta_df.to_dict('records')
        db._fp_matrix = fps

        return db

    def filter_by(
        self,
        direction: str | None = None,
        min_amount_wan: float | None = None,
        min_order_count: int | None = None,
    ) -> 'BehaviorDB':
        """按条件筛选子集（返回新DB，不修改原DB）"""
        sub = BehaviorDB()
        for i in range(len(self)):
            meta = self.metadata[i]
            if direction and meta.get('direction') != direction:
                continue
            if min_amount_wan and meta.get('total_amount_wan', 0) < min_amount_wan:
                continue
            if min_order_count and meta.get('order_count', 0) < min_order_count:
                continue
            sub.add(self.fingerprints[i], self.returns[i], meta)
        return sub
