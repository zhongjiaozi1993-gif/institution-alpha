"""label_builder 前向收益修复的最小回归测试。

保证: 计算远期收益时保留 end_date 之后的行情，年末信号日 label 不被截断为 NaN；
输出仍按信号日过滤到 <= end_date；label_20d 存在且数值正确。
"""
import numpy as np
import pandas as pd

from src.features import label_builder as lb


def _write_daily(tmp_dir, code, dates, opens):
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": np.asarray(opens, dtype=float),
        "high": np.asarray(opens, dtype=float),
        "low": np.asarray(opens, dtype=float),
        "close": np.asarray(opens, dtype=float),
        "volume": 1.0, "amount": 1.0, "outstanding_share": 1.0, "turnover": 1.0,
    })
    df.to_parquet(tmp_dir / f"{code}.parquet", index=False)


def test_forward_labels_not_truncated_at_end_date(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "DAILY_DIR", tmp_path)  # 无 idx 文件 → 不生成超额, 专测原始 label
    # 行情跨年到 2026-02-15，信号窗口截到 2025-12-31
    dates = pd.bdate_range("2025-12-01", "2026-02-15")
    opens = np.arange(1, len(dates) + 1, dtype=float)  # 严格递增, 便于校验数值
    _write_daily(tmp_path, "000001", dates, opens)

    out = lb.build_labels(["000001"], "2025-01-01", "2025-12-31", horizons=[5, 10, 20])
    out["trade_date"] = pd.to_datetime(out["trade_date"])

    # 输出按信号日过滤
    assert out["trade_date"].max() <= pd.Timestamp("2025-12-31")
    # 新增 20d
    assert "label_20d" in out.columns
    # 年末最后一个信号日的远期 label 非 NaN（保留了 end_date 之后的行情）
    last = out.sort_values("trade_date").iloc[-1]
    assert not np.isnan(last["label_10d"])
    assert not np.isnan(last["label_20d"])
    # 数值正确性: opens 递增, 对某信号日 T, entry=open[T+1], exit=open[T+1+h]
    full = pd.DataFrame({"date": dates, "open": opens}).sort_values("date").reset_index(drop=True)
    t0 = out.sort_values("trade_date").iloc[0]["trade_date"]
    i = full.index[full["date"] == t0][0]
    exp5 = full.loc[i + 6, "open"] / full.loc[i + 1, "open"] - 1.0
    got5 = out.sort_values("trade_date").iloc[0]["label_5d"]
    assert abs(got5 - exp5) < 1e-9


def test_truncation_regression_when_no_future_rows(tmp_path, monkeypatch):
    """行情本身止于 end_date（如退市/陈旧缓存）→ 末端远期 label 合理地为 NaN。"""
    monkeypatch.setattr(lb, "DAILY_DIR", tmp_path)
    dates = pd.bdate_range("2025-12-01", "2025-12-31")
    _write_daily(tmp_path, "000002", dates, np.arange(1, len(dates) + 1))
    out = lb.build_labels(["000002"], "2025-01-01", "2025-12-31", horizons=[10])
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    last = out.sort_values("trade_date").iloc[-1]
    assert np.isnan(last["label_10d"])  # 无远期行情 → NaN, 属数据本身
