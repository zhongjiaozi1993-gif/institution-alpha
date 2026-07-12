"""Phase 5.2B purge 切分与 alpha 方向反转判据的回归测试。"""
import numpy as np
import pandas as pd

from src.validation import level2_validator as lv


def _cal():
    return pd.bdate_range("2025-06-02", "2025-10-31").values  # 跨 test_start 的交易日历


def test_purge_label_end_strictly_before_test_start():
    ts = "2025-09-01"
    for h in (5, 10):
        cut, info = lv.purge_split_info(_cal(), "2025-08-31", ts, horizon=h, embargo=6)
        assert cut is not None
        # 核心：末 train 信号日的 label 结束日 **严格早于** 首 test 日 → train 标签不窥探 test
        assert pd.Timestamp(info["last_train_label_end_date"]) < pd.Timestamp(info["first_test_trade_date"])
        assert cut <= pd.Timestamp("2025-08-31")


def test_purge_longer_horizon_cuts_earlier():
    cut5, _ = lv.purge_split_info(_cal(), "2025-08-31", "2025-09-01", 5, 6)
    cut10, _ = lv.purge_split_info(_cal(), "2025-08-31", "2025-09-01", 10, 6)
    assert cut10 < cut5  # 更长 horizon 需要更早的 train 截止


def test_embargo_is_additional_not_replacement():
    # embargo=0 时仅 horizon purge，label_end 仍需 < test_start
    cut0, info0 = lv.purge_split_info(_cal(), "2025-08-31", "2025-09-01", 5, 0)
    assert pd.Timestamp(info0["last_train_label_end_date"]) < pd.Timestamp(info0["first_test_trade_date"])
    # embargo>0 在 horizon purge 之上再往前留空档 → 截止更早
    cut6, _ = lv.purge_split_info(_cal(), "2025-08-31", "2025-09-01", 5, 6)
    assert cut6 < cut0


def test_alpha_direction_flip_rule():
    # sign_a=-1（train 负 IC），test 原始 IC 也为负 → 方向一致，NOT flipped（修复前会误报）
    assert lv.alpha_direction_flipped(-1.0, -0.0751) is False
    assert lv.alpha_direction_flipped(-1.0, 0.05) is True    # test 变正 → 反转
    assert lv.alpha_direction_flipped(1.0, 0.05) is False    # 均正 → 一致
    assert lv.alpha_direction_flipped(1.0, -0.05) is True    # 由正变负 → 反转
    assert lv.alpha_direction_flipped(-1.0, float("nan")) is False
