"""Phase 5.2A — Level-2 目录结构兼容 + 覆盖审计 + 断点续跑 测试。

覆盖 spec 第七节要求的 6 个测试：
  test_resolve_wind_subdir_layout / test_resolve_flat_day_layout
  test_no_silent_skip / test_layouts_produce_same_features
  test_resume_skips_completed_symbol / test_audit_statuses
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data import level2_reader as l2
from src.features import level2_feature_builder as fb
from src.features import level2_batch as batch

TRADE_HEADER = ("万得代码,交易所代码,自然日,时间,成交编号,成交代码,委托代码,BS标志,"
                "成交价格,成交数量,叫卖序号,叫买序号")
ORDER_HEADER = ("万得代码,交易所代码,自然日,时间,委托编号,交易所委托号,委托类型,委托代码,"
                "委托价格,委托数量")


def _write_trades(target: Path, day: str, code: str, header_only: bool = False):
    target.mkdir(parents=True, exist_ok=True)
    lines = [TRADE_HEADER]
    if not header_only:
        lines += [
            f"{code}.SZ,SZ,{day},093000000,1,0,B,B,100000,1000,9001,1001",
            f"{code}.SZ,SZ,{day},100000000,2,0,S,S,101000,2000,1002,9002",
        ]
    (target / "逐笔成交.csv").write_text("\n".join(lines) + "\n", encoding="gb18030")


def _write_orders(target: Path, day: str, code: str):
    target.mkdir(parents=True, exist_ok=True)
    lines = [
        ORDER_HEADER,
        f"{code}.SZ,SZ,{day},093000000,1001,0,A,B,100000,1000",
        f"{code}.SZ,SZ,{day},100000000,1002,0,A,S,101000,2000",
    ]
    (target / "逐笔委托.csv").write_text("\n".join(lines) + "\n", encoding="gb18030")


def _make_day(root: Path, code: str, day: str, layout: str,
              header_only: bool = False, with_orders: bool = True):
    """在 {root}/{code}/raw/{day}/ 下按 layout 写逐笔文件。"""
    day_dir = root / code / "raw" / day
    if layout == "wind_subdir":
        target = day_dir / l2._build_wind_code(code)
    else:
        target = day_dir
    _write_trades(target, day, code, header_only=header_only)
    if with_orders:
        _write_orders(target, day, code)
    return day_dir


# ─────────────────────────────────────────────────────────────
# 1 + 2: resolve 两种结构
# ─────────────────────────────────────────────────────────────
def test_resolve_wind_subdir_layout(tmp_path):
    code = "000001"
    day_dir = _make_day(tmp_path, code, "20250303", "wind_subdir")
    res = fb.resolve_level2_day_dir(code, day_dir)
    assert res.layout_type == "wind_subdir"
    assert res.selected_dir == day_dir / "000001.SZ"
    assert res.has_trade_file is True
    assert res.has_order_file is True


def test_resolve_flat_day_layout(tmp_path):
    code = "000001"
    day_dir = _make_day(tmp_path, code, "20250303", "flat_day_dir", with_orders=False)
    res = fb.resolve_level2_day_dir(code, day_dir)
    assert res.layout_type == "flat_day_dir"
    assert res.selected_dir == day_dir
    assert res.has_trade_file is True
    assert res.has_order_file is False


def test_resolve_no_supported_layout(tmp_path):
    day_dir = tmp_path / "000001" / "raw" / "20250303"
    (day_dir / "000001.SZ").mkdir(parents=True)
    (day_dir / "行情.csv").write_text("x\n", encoding="gb18030")  # 有目录但无逐笔成交
    res = fb.resolve_level2_day_dir("000001", day_dir)
    assert res.layout_type == "no_supported_layout"
    assert res.selected_dir is None
    assert res.has_trade_file is False


# ─────────────────────────────────────────────────────────────
# 3: 不静默跳过 — 每个 symbol-day 都进审计
# ─────────────────────────────────────────────────────────────
def test_no_silent_skip(tmp_path):
    code = "000001"
    _make_day(tmp_path, code, "20250303", "flat_day_dir")                       # ok
    _make_day(tmp_path, code, "20250304", "flat_day_dir", header_only=True)     # empty_trades
    # 20250305: 目录存在但无逐笔成交 → no_supported_layout
    d = tmp_path / code / "raw" / "20250305"
    d.mkdir(parents=True)
    (d / "行情.csv").write_text("x\n", encoding="gb18030")

    df, audit_rows = fb.build_stock_features(code, single_stock_root=tmp_path)
    audit = fb.audit_frame(audit_rows)

    # 3 个日目录 → 恰好 3 条审计，无遗漏
    assert set(audit["day"]) == {"20250303", "20250304", "20250305"}
    assert len(audit) == 3
    assert dict(audit["status"].value_counts()) == {
        "ok": 1, "empty_trades": 1, "no_supported_layout": 1}
    # 只有 ok 日产出特征行
    assert len(df) == 1
    assert df.iloc[0]["trade_date"] == pd.Timestamp("2025-03-03")
    assert df.iloc[0]["layout_type"] == "flat_day_dir"


# ─────────────────────────────────────────────────────────────
# 4: 两种结构相同内容 → 相同特征
# ─────────────────────────────────────────────────────────────
def test_layouts_produce_same_features(tmp_path):
    code = "000001"
    root_wind = tmp_path / "wind"
    root_flat = tmp_path / "flat"
    _make_day(root_wind, code, "20250303", "wind_subdir")
    _make_day(root_flat, code, "20250303", "flat_day_dir")

    df_w, _ = fb.build_stock_features(code, single_stock_root=root_wind)
    df_f, _ = fb.build_stock_features(code, single_stock_root=root_flat)

    assert len(df_w) == 1 and len(df_f) == 1
    assert df_w.iloc[0]["layout_type"] == "wind_subdir"
    assert df_f.iloc[0]["layout_type"] == "flat_day_dir"
    # 相同逐笔内容 → 全部 l2_* 特征一致
    pd.testing.assert_series_equal(
        df_w[fb.FEATURE_NAMES].iloc[0], df_f[fb.FEATURE_NAMES].iloc[0],
        check_names=False)


# ─────────────────────────────────────────────────────────────
# 5: resume 跳过已完成股票
# ─────────────────────────────────────────────────────────────
def test_resume_skips_completed_symbol(tmp_path):
    code = "000001"
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    _make_day(data_root, code, "20250303", "flat_day_dir")

    batch.run(data_root=data_root, output_root=out_root, workers=1)

    assert batch.is_symbol_complete(out_root, code, fb.FEATURE_VERSION) is True
    # resume: 已完成 → 不再处理
    assert batch.select_symbols([code], out_root, fb.FEATURE_VERSION,
                                resume=True, force=False, limit=None) == []
    # force: 强制重算
    assert batch.select_symbols([code], out_root, fb.FEATURE_VERSION,
                                resume=True, force=True, limit=None) == [code]
    # 版本不一致 → 视为未完成
    assert batch.select_symbols([code], out_root, "vX",
                                resume=True, force=False, limit=None) == [code]


# ─────────────────────────────────────────────────────────────
# 6: 五种审计状态齐全
# ─────────────────────────────────────────────────────────────
def test_audit_statuses(tmp_path, monkeypatch):
    code = "000001"
    # 每天都放一个逐笔成交文件（除 no_supported 日），让 resolver 选中 flat；
    # reader 由 monkeypatch 决定 ok / empty / decode / other。
    for day in ("20250303", "20250304", "20250306", "20250307"):
        _make_day(tmp_path, code, day, "flat_day_dir")
    d = tmp_path / code / "raw" / "20250305"          # no_supported_layout
    d.mkdir(parents=True)
    (d / "行情.csv").write_text("x\n", encoding="gb18030")

    ok_cj = pd.DataFrame({
        "时间": ["093000000"], "成交价格": [100000], "成交数量": [1000.0],
        "成交金额": [100000 * 1000], "BS标志": ["B"]})

    def fake_read(selected_dir):
        s = str(selected_dir)
        if "20250303" in s:
            return {"逐笔成交": ok_cj.copy(), "逐笔委托": pd.DataFrame()}
        if "20250304" in s:
            return {"逐笔成交": pd.DataFrame(), "逐笔委托": pd.DataFrame()}
        if "20250306" in s:
            raise UnicodeDecodeError("gb18030", b"", 0, 1, "bad byte")
        if "20250307" in s:
            raise ValueError("synthetic parser failure")
        raise AssertionError(f"unexpected dir {s}")

    monkeypatch.setattr(fb.l2, "read_level2_stock_dir", fake_read)

    df, audit_rows = fb.build_stock_features(code, single_stock_root=tmp_path)
    audit = fb.audit_frame(audit_rows)

    status_by_day = dict(zip(audit["day"], audit["status"]))
    assert status_by_day == {
        "20250303": "ok",
        "20250304": "empty_trades",
        "20250305": "no_supported_layout",
        "20250306": "decode_error",
        "20250307": "other_error",
    }
    assert set(audit["status"]) == {
        "ok", "empty_trades", "no_supported_layout", "decode_error", "other_error"}
    assert len(df) == 1  # 仅 ok 日
