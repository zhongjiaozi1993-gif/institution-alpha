"""
institution_profiles.py — 机构深度画像系统

对出现在002516股东名单中的机构建立独立画像，追踪投资行为和动机。

覆盖维度：
  1. 机构身份：类型、管理人、委托人/持有人结构
  2. 持仓分析：历史持仓变化、进入/退出时机、成本估算
  3. 投资动机：为什么买这只股票，看中什么逻辑
  4. 业绩追踪：净值/产品表现
  5. 关联网络：同一管理人/委托人的其他可查持仓

机构分为四层画像深度：
  L1 — 身份识别 + 持仓时间线
  L2 — L1 + 投资逻辑/动机分析
  L3 — L2 + 关联网络/同系产品
  L4 — L3 + 实时追踪信号（后续Phase）
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
STOCK = "002516"
STOCK_NAME = "旷达科技"
EVIDENCE_DIR = PROJECT / "data" / "single_stock" / STOCK / "evidence"
PRICE_PATH = PROJECT / "data" / "single_stock" / STOCK / "price_daily.csv"
OUTPUT_DIR = PROJECT / "data" / "single_stock" / STOCK / "profiles"

EASTMONEY_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 0: 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_evidence_data() -> dict[str, pd.DataFrame]:
    """Load all evidence chain outputs."""
    result = {}
    for name in ["public_evidence", "holder_changes", "daily_evidence",
                 "notable_events", "source_status"]:
        path = EVIDENCE_DIR / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(path)
        else:
            result[name] = pd.DataFrame()
    return result


def load_price_data() -> pd.DataFrame:
    """Load daily price data for cost estimation."""
    prices = pd.read_csv(PRICE_PATH)
    prices["日期"] = pd.to_datetime(prices["日期"])
    prices = prices.sort_values("日期")
    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: 机构识别与分类
# ═══════════════════════════════════════════════════════════════════════════════

def extract_institutions(public_evidence: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique institution names from public evidence,
    classified by type and tracked across time.
    """
    sources = ["sina_fund_holders", "sina_main_holders", "sina_circulate_holders"]
    holders = public_evidence[public_evidence["source"].isin(sources)].copy()

    records = []
    for _, row in holders.iterrows():
        try:
            raw = json.loads(row.get("raw", "{}"))
        except Exception:
            continue

        source = row["source"]
        date = str(row.get("date", ""))

        if source == "sina_fund_holders":
            name = raw.get("基金名称", "")
            shares = raw.get("持仓数量", 0)
            ratio = raw.get("占流通股比例", 0)
            inst_type = classify_institution(name, "基金")
        elif source in ("sina_main_holders", "sina_circulate_holders"):
            name = raw.get("股东名称", "")
            shares = raw.get("持股数量", 0)
            if source == "sina_main_holders":
                ratio = raw.get("持股比例", 0)
            else:
                ratio = raw.get("占流通股比例", 0)
            inst_type = classify_institution(name, "股东")

        if not name:
            continue

        records.append({
            "name": str(name).strip(),
            "date": date,
            "source": source,
            "shares": pd.to_numeric(shares, errors="coerce"),
            "ratio": pd.to_numeric(ratio, errors="coerce"),
            "type": inst_type["category"],
            "subtype": inst_type["subtype"],
            "priority": inst_type["priority"],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "name"])

    # Deduplicate: same name+date → keep max shares
    df = df.sort_values("shares", ascending=False).drop_duplicates(
        ["name", "date", "type"], keep="first"
    ).sort_values(["name", "date"]).reset_index(drop=True)

    return df


def classify_institution(name: str, role: str) -> dict:
    """
    Classify institution type for profiling depth.

    Priority 1 = must deep-profile (L3+)
    Priority 2 = profile with available data (L2)
    Priority 3 = statistical tracking only (L1)
    """
    name_clean = str(name).strip()

    # === Priority 1: Strategic/Active institutions ===
    active_keywords = [
        "野村", "日出东方",        # Nomura SMA
        "启创一号",                # PE/VC consortium
    ]
    for kw in active_keywords:
        if kw in name_clean:
            return {"category": "机构产品", "subtype": "专户/私募", "priority": 1}

    if "香港中央结算" in name_clean:
        return {"category": "外资", "subtype": "北向资金", "priority": 1}

    # Major shareholders with active changes
    if any(kw in name_clean for kw in ["沈介良", "旷达控股", "旷达创业"]):
        return {"category": "产业资本", "subtype": "实控人/控股", "priority": 1}

    # === Priority 2: Institutional investors ===
    if "常州产业投资" in name_clean:
        return {"category": "国有资本", "subtype": "地方国资", "priority": 2}

    if "上海纺织" in name_clean:
        return {"category": "国有资本", "subtype": "产业集团", "priority": 2}

    if role == "基金":
        return {"category": "机构产品", "subtype": "公募基金", "priority": 2}

    if any(kw in name_clean for kw in ["信托", "资管", "集合"]):
        return {"category": "机构产品", "subtype": "信托/资管", "priority": 2}

    if "员工持股" in name_clean:
        return {"category": "内部人", "subtype": "员工持股计划", "priority": 2}

    # === Priority 3: Individuals / passive ===
    if any(kw in name_clean for kw in ["中央汇金", "证金"]):
        return {"category": "国有资本", "subtype": "国家队", "priority": 3}

    if "指数" in name_clean:
        return {"category": "机构产品", "subtype": "被动指数基金", "priority": 3}

    return {"category": "个人", "subtype": "自然人", "priority": 3}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: 野村东方日出东方1号 — 深度画像
# ═══════════════════════════════════════════════════════════════════════════════

def profile_nomura_sunrise(institutions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """
    Deep profile of 野村东方国际日出东方1号单一资产管理计划.

    Structure:
      - 产品身份：管理人、委托人、产品类型
      - 持仓历史：002516完整时间线
      - 投资动机：产业协同、母公司关系
      - 委托人分析：上海纺织集团的战略意图
      - 管理人分析：野村东方国际证券的A股业务
    """
    nomura = institutions[institutions["name"].str.contains("野村", na=False)].copy()
    profile = {
        "profile_id": "NOMURA-SUNRISE-001",
        "name": "野村东方国际证券－上海纺织(集团)有限公司－野村东方国际日出东方1号单一资产管理计划",
        "short_name": "野村东方日出东方1号",
        "profile_level": "L3",
    }

    # --- 产品身份 ---
    profile["identity"] = {
        "管理人": "野村东方国际证券有限公司",
        "管理人类型": "外资控股券商（野村控股51%+东方国际集团49%）",
        "委托人": "上海纺织(集团)有限公司",
        "产品类型": "单一资产管理计划（一对一专户）",
        "产品特征": "非公开募集，一对一委托，委托人主导投资方向",
        "成立时间_推测": "2021年前（002516持仓最早可查2021Q4）",
    }

    # --- 002516持仓历史 ---
    if not nomura.empty:
        nomura_sorted = nomura.sort_values("date")
        first_date = nomura_sorted["date"].iloc[0]
        last_date = nomura_sorted["date"].iloc[-1]
        shares_latest = nomura_sorted["shares"].iloc[-1]
        ratio_latest = nomura_sorted["ratio"].iloc[-1]

        # Check if shares ever changed
        unique_shares = nomura["shares"].dropna().unique()
        shares_changed = len(unique_shares) > 1

        profile["holdings_002516"] = {
            "首次出现": first_date.strftime("%Y-%m-%d"),
            "最新记录": last_date.strftime("%Y-%m-%d"),
            "持仓数量_股": int(shares_latest),
            "持仓数量_万股": round(shares_latest / 10000, 0),
            "占流通股比例": f"{ratio_latest:.2f}%" if pd.notna(ratio_latest) else "N/A",
            "持仓变动": "从未变动" if not shares_changed else f"历经{len(unique_shares)}种持仓量",
            "持仓时长": f"{(last_date - first_date).days}天 ({(last_date - first_date).days/365:.1f}年)",
            "定性": "长期锚定持仓，暂未观察到交易型调仓",
        }

        # Cost estimation
        profile["cost_estimation"] = estimate_entry_cost(nomura, prices)

    # --- 投资动机分析 ---
    profile["motivation"] = analyze_nomura_motivation()

    # --- 委托人分析 ---
    profile["client_analysis"] = analyze_shanghai_textile()

    # --- 管理人分析 ---
    profile["manager_analysis"] = analyze_nomura_orient_international()

    return profile


def estimate_entry_cost(nomura_holdings: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Estimate cost basis of 野村's 002516 position."""
    first_date = nomura_holdings["date"].min()
    # Look at price range in the quarter before first appearance
    quarter_start = first_date - pd.DateOffset(months=3)
    price_window = prices[
        (prices["日期"] >= quarter_start) & (prices["日期"] <= first_date)
    ]

    if price_window.empty:
        # Try wider window
        price_window = prices[prices["日期"] <= first_date].tail(60)

    if price_window.empty:
        return {"状态": "无法估算（缺少早期价格数据）"}

    # 后复权价格
    close_col = "收盘" if "收盘" in price_window.columns else "close"
    if close_col not in price_window.columns:
        return {"状态": "价格列名不匹配"}

    prices_clean = pd.to_numeric(price_window[close_col], errors="coerce").dropna()
    if prices_clean.empty:
        return {"状态": "价格数据为空"}

    avg_price = prices_clean.mean()
    min_price = prices_clean.min()
    max_price = prices_clean.max()

    # 后复权价格需要折算到实际交易价
    # 002516在2021年后的复权因子大约7-8倍
    latest_price = pd.to_numeric(
        prices[prices["日期"] == prices["日期"].max()][close_col].values[0]
        if close_col in prices.columns else 0,
        errors="coerce"
    )

    return {
        "建仓期间_均价_后复权": f"{avg_price:.2f}元",
        "建仓期间_最低_后复权": f"{min_price:.2f}元",
        "建仓期间_最高_后复权": f"{max_price:.2f}元",
        "估算建仓期间": f"{quarter_start.strftime('%Y-%m')} ~ {first_date.strftime('%Y-%m')}",
        "note": "后复权价格含历史分红拆细调整，实际交易价需折算",
    }


def analyze_nomura_motivation() -> dict:
    """
    Analyze why 野村东方日出东方1号 bought 旷达科技.

    关键线索（注意：以下为证据链推断，不是已穿透披露事实）：
    1. 这是一对一专户，委托人上海纺织集团主导投资决策
    2. 上海纺织集团是东方国际集团的核心子公司
    3. 野村东方国际证券的股东之一就是东方国际集团（占49%）
    4. 所以这是「大股东关联方委托券商子公司管理资金→投向指定标的」的闭环

    投资逻辑链：
      上海纺织集团（有钱，想投资）
        → 委托野村东方国际证券（东方国际占股49%，自家人）
          → 成立一对一专户「日出东方1号」
            → 买入旷达科技1200万股（长期持有）
    """
    return {
        "核心逻辑": "长期定向资管持仓；产业协同是高价值假设，仍需公告/年报原文核实",
        "驱动因素": [
            {
                "因素": "关联方闭环（待核实）",
                "分析": (
                    "股东名称显示产品链条包含野村东方国际证券、上海纺织(集团)有限公司和"
                    "日出东方1号单一资产管理计划。若上海纺织与东方国际集团/野村东方的"
                    "股权关系属实，则可能形成委托人、管理人、股东方的关联闭环。"
                    "当前脚本将其标记为强推断，不作为已证实事实。"
                ),
            },
            {
                "因素": "汽车内饰产业链协同（动机假设）",
                "分析": (
                    "旷达科技主营汽车内饰面料，是国内该细分领域龙头。"
                    "上海纺织集团是纺织业巨头，在面料、新材料领域有深厚积累。"
                    "两者在汽车内饰材料产业链上有天然协同——"
                    "这种持仓可能是产业联盟或产业配置的表达。"
                    "但目前还没有抓到直接写明双方战略合作的公告。"
                ),
            },
            {
                "因素": "长期持有验证",
                "分析": (
                    "1200万股从2021年Q4持有至今（2026Q1），穿越牛熊从未交易。"
                    "这支持'长期锚定资金'判断，但不能直接证明其拥有董事会观察席或合作协议。"
                ),
            },
            {
                "因素": "市值与仓位匹配",
                "分析": (
                    "1200万股 × 旷达科技均价约6元 = 约7200万元。"
                    "对于一个专户产品来说，单票仓位可能占比较高（10-30%），"
                    "说明委托人对此标的的信心较强，可能是'核心持仓'定位。"
                ),
            },
        ],
        "投资风格": "长期锚定持有；是否产业逻辑驱动仍需继续核实",
        "可能触发事件": [
            "上海纺织与旷达科技是否存在战略合作协议（需核实公告）",
            "东方国际集团/上海纺织的产业布局调整",
            "旷达科技定向增发或其他再融资（专户可能参与）",
        ],
    }


def analyze_shanghai_textile() -> dict:
    """Profile the client: 上海纺织(集团)有限公司."""
    return {
        "全称": "上海纺织(集团)有限公司",
        "证据状态": "待核实：以下为公开常识/名称链条推断，需用工商、年报或产品备案补证。",
        "实控人": "待核实：可能与东方国际集团/上海国资体系相关",
        "行业": "纺织业/产业投资（待核实）",
        "总资产": "待核实",
        "核心业务": [
            "纺织品生产与贸易",
            "汽车内饰材料（与旷达科技同赛道）",
            "时尚产业运营",
            "产业投资与资本运营",
        ],
        "关联上市公司": [
            "龙头股份(600630) — 东方国际集团控股",
            "申达股份(600626) — 东方国际集团控股",
            "东方创业(600278) — 东方国际集团控股",
        ],
        "002516关系分析": (
            "从产品名称看，上海纺织是该单一资管计划的重要相关方。"
            "旷达科技汽车内饰面料业务与纺织/新材料存在产业链交集，"
            "可能构成持仓动机之一；但尚未抓到双方战略合作公告，"
            "因此只能作为动机假设。"
        ),
    }


def analyze_nomura_orient_international() -> dict:
    """Profile the manager: 野村东方国际证券."""
    return {
        "全称": "野村东方国际证券有限公司",
        "成立": "2019年（待官网/监管资料核实）",
        "股东结构": "待核实：脚本暂按公开资料线索记录，需补官网/监管来源",
        "注册地": "上海",
        "业务范围": [
            "证券经纪",
            "证券投资咨询",
            "证券资产管理（含单一资管计划）",
            "证券承销与保荐",
        ],
        "母公司野村控股": {
            "性质": "日本最大券商，东京交易所上市(8604.T)",
            "AUM": "约50万亿日元（全球）",
            "中国战略": "通过野村东方国际拓展中国资管和投行业务",
        },
        "日出东方系列产品": (
            "'日出东方1号'出现在股东名称中，说明这是单一资产管理计划。"
            "单一资管计划对接单一委托人，灵活性高，"
            "但具体投资范围、委托人权利和产品合同目前未公开取得。"
        ),
        "关键观察": (
            "公开持仓能证明该产品长期持有002516 1200万股；"
            "产品名称能提示野村东方国际证券与上海纺织相关。"
            "但最终出资人、产品合同、投资决策权归属未穿透披露，"
            "因此只能将其定性为'长期锚定资管账户'，不能直接说成自家资金通道。"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: 其他重点机构画像
# ═══════════════════════════════════════════════════════════════════════════════

def profile_shen_jieliang(institutions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Profile 沈介良 (actual controller, major shareholder)."""
    holder = institutions[institutions["name"].str.contains("沈介良", na=False)].copy()
    profile = {
        "profile_id": "SHENJIELIANG-001",
        "name": "沈介良",
        "short_name": "沈介良(实控人)",
        "profile_level": "L3",
        "identity": {
            "身份": "旷达科技实际控制人、创始人、董事长",
            "持股特征": "绝对控股→大幅减持→仍为第一大股东",
        },
    }

    if not holder.empty:
        holder_sorted = holder.sort_values("date")
        shares_history = holder_sorted[["date", "shares", "ratio"]].dropna()
        shares_history["shares_wan"] = shares_history["shares"] / 10000

        first = shares_history.iloc[0]
        last = shares_history.iloc[-1]

        # Detect the massive transfer
        max_shares = shares_history["shares"].max()
        min_shares = shares_history["shares"].min()
        delta = min_shares - max_shares

        profile["holdings_002516"] = {
            "最早记录": first["date"].strftime("%Y-%m-%d"),
            "最初持股_万股": round(first["shares"] / 10000, 0),
            "最高持股_万股": round(max_shares / 10000, 0),
            "最新持股_万股": round(last["shares"] / 10000, 0),
            "最新占比": f"{last['ratio']:.2f}%" if pd.notna(last['ratio']) else "N/A",
            "变动": f"{'增持' if delta > 0 else '减持'}{abs(delta)/10000:.0f}万股",
            "重大事件": None,
        }

        # Detect the 4.1亿股转让事件 (2025 Q4)
        # Only look at DEcreases (transfers out), not increases (stock splits)
        share_diffs = shares_history.set_index("date")["shares"].diff()
        decreases = share_diffs[share_diffs < -50_000_000]  # >5000万股减少
        if len(decreases) > 0:
            event_date = decreases.index[0]
            before_shares = float(shares_history[
                shares_history["date"] < event_date
            ]["shares"].iloc[-1])
            after_shares = float(shares_history[
                shares_history["date"] >= event_date
            ]["shares"].iloc[0])
            transferred = before_shares - after_shares

            profile["holdings_002516"]["重大事件"] = {
                "时间": event_date.strftime("%Y-%m-%d"),
                "事件": f"转让{transferred/10000:.0f}万股（{transferred/100000000:.1f}亿股）",
                "受让方": "株洲启创一号产业投资合伙企业(有限合伙)",
                "转让比例": f"{transferred/before_shares*100:.1f}%（占转让前持股）",
                "转让后剩余": f"{after_shares/10000:.0f}万股（{after_shares/before_shares*100:.1f}%）",
                "可能原因": [
                    "家族财富规划/资产隔离",
                    "引入战略投资者（国资背景产业基金）",
                    "税务筹划（通过合伙企业架构）",
                    f"减持套现（转让价估算：{transferred/10000:.0f}万股×~5元=~{transferred/10000*5/10000:.0f}亿元）",
                ],
            }

    profile["motivation"] = {
        "减持动机": (
            "沈介良从46.63%减持到18.63%（转让28%给启创一号），"
            "但仍为第一大股东和控制人。这种大幅减持但不失控股权的方式，"
            "通常意味着：1）个人财富变现需求；2）引入战略投资者优化股东结构；"
            "3）通过转让给PE基金为后续资本运作做准备。"
        ),
        "剩余持股意义": (
            "仍持有18.63%，价值约15亿（按后复权40-50元/股计算市值）。"
            "保留控制权说明对公司未来仍有信心。"
        ),
    }

    return profile


def profile_qichuang_one(institutions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Profile 株洲启创一号产业投资合伙企业(有限合伙)."""
    holder = institutions[institutions["name"].str.contains("启创一号", na=False)].copy()
    profile = {
        "profile_id": "QICHUANG-001",
        "name": "株洲启创一号产业投资合伙企业(有限合伙)",
        "short_name": "启创一号(产业基金)",
        "profile_level": "L2",
        "identity": {
            "类型": "有限合伙制产业投资基金",
            "注册地": "湖南株洲",
            "GP推测": "株洲当地国资背景的基金管理人",
            "LP可能来源": "株洲市产业发展引导基金、社会资本",
            "与沈介良关系": "2025年Q3/Q4受让沈介良28%股份",
        },
    }

    if not holder.empty:
        holder_sorted = holder.sort_values("date")
        profile["holdings_002516"] = {
            "持股数量": f"{holder_sorted['shares'].iloc[-1]/10000:.0f}万股",
            "持股比例": f"{holder_sorted['ratio'].iloc[-1]:.2f}%",
            "进入方式": "协议受让（沈介良转让）",
            "进入时间": holder_sorted["date"].iloc[0].strftime("%Y-%m-%d"),
        }

    profile["motivation"] = {
        "投资逻辑": [
            "承接大股东减持，可能是战略投资者角色",
            "株洲是汽车产业重镇（中车/北汽株洲基地），与旷达汽车内饰业务协同",
            "产业基金通常有3-7年退出规划，不会短期减持",
            "通过合伙企业形式规避直接持股的税务和披露义务",
        ],
        "风险提示": [
            "LP退出期临近时可能要求减持",
            "基金到期（一般5-7年）后需要退出安排",
            "如果旷达不在株洲布局产业，协同效应有限",
        ],
    }

    return profile


def profile_northbound(institutions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Profile 香港中央结算有限公司 (Northbound/沪深股通)."""
    holder = institutions[
        institutions["name"].str.contains("香港中央结算", na=False)
    ].copy()
    profile = {
        "profile_id": "NORTHBOUND-001",
        "name": "香港中央结算有限公司",
        "short_name": "北向资金",
        "profile_level": "L2",
        "identity": {
            "性质": "沪深港通北向资金的合计名义持有人",
            "实质": "代表所有通过沪/深股通买入A股的境外投资者",
            "无法穿透": "不能直接知道背后是哪些外资机构",
            "参考意义": "北向资金整体流向反映外资对该股的看法",
        },
    }

    if not holder.empty:
        holder_sorted = holder.sort_values("date")
        shares_ts = holder_sorted.set_index("date")["shares"].dropna()

        if len(shares_ts) > 1:
            first = shares_ts.iloc[0]
            last = shares_ts.iloc[-1]
            max_s = shares_ts.max()
            min_s = shares_ts.min()
            trend = "增持" if last > first else "减持"

            profile["holdings_002516"] = {
                "首次记录": f"{shares_ts.index[0].strftime('%Y-%m-%d')}, {first/10000:.0f}万股",
                "最新记录": f"{shares_ts.index[-1].strftime('%Y-%m-%d')}, {last/10000:.0f}万股",
                "最高": f"{shares_ts.idxmax().strftime('%Y-%m-%d')}, {max_s/10000:.0f}万股",
                "最低": f"{shares_ts.idxmin().strftime('%Y-%m-%d')}, {min_s/10000:.0f}万股",
                "趋势": trend,
                "波动率": f"{shares_ts.std()/shares_ts.mean()*100:.1f}%",
            }

            # Trading activity analysis
            if len(shares_ts) >= 4:
                changes = shares_ts.diff().dropna()
                active_quarters = (changes.abs() > changes.std()).sum()
                profile["trading_pattern"] = {
                    "活跃度": "高频交易" if active_quarters >= len(changes) * 0.5 else "低频调仓",
                    "增持季度": f"{int((changes > 0).sum())}个",
                    "减持季度": f"{int((changes < 0).sum())}个",
                    "最大单季增持": f"{changes.max()/10000:.0f}万股" if changes.max() > 0 else "N/A",
                    "最大单季减持": f"{abs(changes.min())/10000:.0f}万股" if changes.min() < 0 else "N/A",
                }

    return profile


def profile_changzhou_capital(institutions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Profile 常州产业投资集团有限公司."""
    holder = institutions[institutions["name"].str.contains("常州产业投资", na=False)].copy()
    profile = {
        "profile_id": "CHANGZHOU-CAPITAL-001",
        "name": "常州产业投资集团有限公司",
        "short_name": "常州产投(地方国资)",
        "profile_level": "L2",
        "identity": {
            "性质": "常州市属国有资本投资运营平台",
            "实控人": "常州市国资委",
            "定位": "地方产业引导和股权投资",
        },
    }

    if not holder.empty:
        holder_sorted = holder.sort_values("date")
        last = holder_sorted.iloc[-1]
        first = holder_sorted.iloc[0]
        profile["holdings_002516"] = {
            "持股数量": f"{last['shares']/10000:.0f}万股" if pd.notna(last['shares']) else "N/A",
            "最新占比": f"{last['ratio']:.2f}%" if pd.notna(last['ratio']) else "N/A",
            "首次出现": first["date"].strftime("%Y-%m-%d"),
            "变动": "长期稳定持仓" if len(holder_sorted["shares"].unique()) <= 2 else "有调仓",
        }

    profile["motivation"] = {
        "投资逻辑": (
            "旷达科技总部在常州，是当地重要上市公司。"
            "常州产投持股是地方国资支持本地企业的标准操作。"
            "这种持股通常不会交易，是'压舱石'型股东。"
        ),
    }

    return profile


def profile_index_funds(institutions: pd.DataFrame) -> list[dict]:
    """Profile major index fund holders."""
    index_funds = institutions[
        institutions["name"].str.contains("指数|ETF", na=False)
    ].copy()

    profiles = []
    # Group by fund family
    if not index_funds.empty:
        index_funds["fund_family"] = index_funds["name"].str.extract(
            r"^(.*?)(?:中证|沪深|国证|上证|指数)"
        )[0].fillna("其他")

        families = index_funds.groupby("fund_family")
        for family, group in families:
            if len(group) < 3:
                continue
            latest = group.sort_values("date").iloc[-1]
            total_shares = group[group["date"] == group["date"].max()]["shares"].sum()

            profiles.append({
                "profile_id": f"INDEX-{family[:8]}",
                "name": f"{family}系列指数基金",
                "short_name": f"{family}(指数)",
                "profile_level": "L1",
                "type": "被动指数基金",
                "latest_holdings": {
                    "日期": latest["date"].strftime("%Y-%m-%d"),
                    "合计持仓_万股": f"{total_shares/10000:.1f}",
                    "产品数量": f"{len(group)}只",
                },
                "motivation": {
                    "逻辑": "被动指数配置，非主动选股。纳入中证1000/2000等小盘指数后自动买入。",
                    "信号意义": "产品数量增加=该股在更多指数中被纳入，反映市值/流动性提升",
                },
            })

    return profiles


def profile_employees(institutions: pd.DataFrame) -> dict | None:
    """Profile 员工持股计划."""
    emp = institutions[institutions["name"].str.contains("员工持股", na=False)]
    if emp.empty:
        return None

    latest = emp.sort_values("date").iloc[-1]
    return {
        "profile_id": "ESOP-001",
        "name": "旷达科技集团股份有限公司－2024年员工持股计划",
        "short_name": "员工持股计划",
        "profile_level": "L1",
        "identity": {
            "类型": "员工持股计划",
            "年份": "2024年",
            "意义": "管理层和核心员工利益绑定",
        },
        "holdings_002516": {
            "持股数量": f"{latest['shares']/10000:.0f}万股",
            "占比": f"{latest['ratio']:.2f}%",
        },
    }


def profile_kuangda_holding(institutions: pd.DataFrame) -> dict:
    """Profile 旷达控股集团有限公司 (Shen's holding company)."""
    holder = institutions[institutions["name"].str.contains("旷达控股集团", na=False)]
    profile = {
        "profile_id": "KUANGDA-HOLDING-001",
        "name": "旷达控股集团有限公司",
        "short_name": "旷达控股(实控人平台)",
        "profile_level": "L2",
        "identity": {
            "性质": "沈介良个人控股平台",
            "与沈介良关系": "沈介良通过旷达控股间接持有旷达科技股份",
            "角色": "实控人顶层持股架构",
        },
    }
    if not holder.empty:
        latest = holder.sort_values("date").iloc[-1]
        first = holder.sort_values("date").iloc[0]
        profile["holdings_002516"] = {
            "持股数量": f"{latest['shares']/10000:.0f}万股",
            "最新占比": f"{latest['ratio']:.2f}%",
            "首次出现": first["date"].strftime("%Y-%m-%d"),
            "趋势": "长期稳定" if len(holder["shares"].unique()) <= 2 else "有变动",
        }
    profile["motivation"] = {
        "逻辑": "沈介良的控股平台，与沈介良直接持股合计构成实际控制权。通常不会交易。",
    }
    return profile


def profile_kuangda_venture(institutions: pd.DataFrame) -> dict:
    """Profile 江苏旷达创业投资有限公司 (Kuangda's PE/VC arm)."""
    holder = institutions[institutions["name"].str.contains("旷达创业投资", na=False)]
    profile = {
        "profile_id": "KUANGDA-VC-001",
        "name": "江苏旷达创业投资有限公司",
        "short_name": "旷达创投(系内PE)",
        "profile_level": "L2",
        "identity": {
            "性质": "旷达科技体系内的产业投资/PE平台",
            "与上市公司关系": "同一实际控制人（沈介良）",
            "可能职能": ["产业链上下游投资", "新业务孵化", "市值管理配合"],
        },
    }
    if not holder.empty:
        latest = holder.sort_values("date").iloc[-1]
        first = holder.sort_values("date").iloc[0]
        profile["holdings_002516"] = {
            "持股数量": f"{latest['shares']/10000:.0f}万股",
            "最新占比": f"{latest['ratio']:.2f}%",
            "首次出现": first["date"].strftime("%Y-%m-%d"),
            "变动": "关注是否跟随沈介良减持" if len(holder) > 1 else "稳定",
        }
    profile["motivation"] = {
        "逻辑": (
            "作为实控人体系内的投资平台，旷达创投的持股变化往往领先于实控人的重大动作。"
            "如果旷达创投开始减持，可能是实控人整体退出的前兆。"
            "目前持股稳定，未跟随沈介良2025Q4的转让。"
        ),
    }
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: 公开信息补充抓取
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_nomura_news() -> list[dict]:
    """Search for news about 野村东方国际日出东方1号."""
    # This is a SMA product, unlikely to have public news
    # But we can check for 野村东方国际证券 news
    results = []
    try:
        url = (
            "https://search-api-web.eastmoney.com/search/jsonp?"
            + urllib.parse.urlencode({
                "cb": "jQuery",
                "param": json.dumps({
                    "uid": "",
                    "keyword": "野村东方国际证券",
                    "type": ["cmsArticleWebOld"],
                    "client": "web",
                    "clientType": "web",
                    "clientVersion": "curr",
                    "param": {
                        "cmsArticleWebOld": {
                            "searchScope": "default",
                            "sort": "default",
                            "pageIndex": 1,
                            "pageSize": 10,
                            "preTag": "<em>",
                            "postTag": "</em>",
                        }
                    },
                }),
                "_": "",
            })
        )
        req = urllib.request.Request(url, headers={"User-Agent": EASTMONEY_UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        text = text[len("jQuery("): -1]
        payload = json.loads(text)
        rows = ((payload.get("result") or {}).get("cmsArticleWebOld")) or []
        for row in rows[:5]:
            results.append({
                "title": row.get("title", ""),
                "date": row.get("date", ""),
                "url": f"http://finance.eastmoney.com/a/{row.get('code', '')}.html",
            })
    except Exception:
        pass
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5: 报告生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(all_profiles: list[dict], stock_code: str,
                    stock_name: str, output_dir: Path) -> str:
    """Generate a complete markdown report from all institution profiles."""
    lines = []
    lines.append(f"# {stock_code} {stock_name} — 机构深度画像报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n画像机构数: {len(all_profiles)}")
    lines.append("")

    # Split into deep and index profiles
    deep = [p for p in all_profiles if p.get("profile_level") in ("L2", "L3")]
    index_funds = [p for p in all_profiles if p.get("profile_level") == "L1"]

    # Table of contents
    lines.append("## 画像目录")
    lines.append("")
    lines.append("### 重点机构 (L2-L3深度画像)")
    for i, p in enumerate(deep):
        level = p.get("profile_level", "")
        name = p.get("short_name", p.get("name", "Unknown"))
        lines.append(f"{i+1}. [{level}] {name}")
    lines.append(f"\n### 指数基金群 (L1统计追踪) — {len(index_funds)} 家基金公司")
    lines.append("")

    # Detailed profiles for L2/L3 only
    for p in deep:
        lines.append("---")
        lines.append(f"## {p.get('profile_level', 'L1')} | {p.get('short_name', p.get('name', ''))}")
        lines.append("")
        lines.append(f"**完整名称**: {p.get('name', '')}")
        lines.append(f"**画像ID**: {p.get('profile_id', '')}")
        lines.append("")

        # Identity section
        if "identity" in p:
            lines.append("### 机构身份")
            lines.append("")
            for k, v in p["identity"].items():
                if isinstance(v, list):
                    lines.append(f"- **{k}**:")
                    for item in v:
                        lines.append(f"  - {item}")
                elif isinstance(v, dict):
                    lines.append(f"- **{k}**:")
                    for sk, sv in v.items():
                        lines.append(f"  - {sk}: {sv}")
                else:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        # Holdings section
        if "holdings_002516" in p:
            lines.append("### 002516持仓")
            lines.append("")
            _dict_to_md(lines, p["holdings_002516"])
            lines.append("")

        # Cost estimation
        if "cost_estimation" in p:
            lines.append("### 成本估算")
            lines.append("")
            _dict_to_md(lines, p["cost_estimation"])
            lines.append("")

        # Trading pattern
        if "trading_pattern" in p:
            lines.append("### 交易模式")
            lines.append("")
            _dict_to_md(lines, p["trading_pattern"])
            lines.append("")

        # Motivation
        if "motivation" in p:
            lines.append("### 投资动机与逻辑")
            lines.append("")
            mot = p["motivation"]
            if isinstance(mot, dict):
                for k, v in mot.items():
                    if isinstance(v, list):
                        lines.append(f"**{k}**:")
                        for item in v:
                            if isinstance(item, dict):
                                lines.append(f"- **{item.get('因素', '')}**: {item.get('分析', '')}")
                            else:
                                lines.append(f"- {item}")
                    elif isinstance(v, dict):
                        lines.append(f"**{k}**:")
                        for sk, sv in v.items():
                            lines.append(f"- {sk}: {sv}")
                    else:
                        lines.append(f"- **{k}**: {v}")
            lines.append("")

        # Client analysis
        if "client_analysis" in p:
            lines.append("### 委托人分析")
            lines.append("")
            _dict_to_md(lines, p["client_analysis"])
            lines.append("")

        # Manager analysis
        if "manager_analysis" in p:
            lines.append("### 管理人分析")
            lines.append("")
            _dict_to_md(lines, p["manager_analysis"])
            lines.append("")

    # L1 Index fund summary
    lines.append("---")
    lines.append("## L1 | 指数基金群 — 统计追踪")
    lines.append("")
    lines.append(f"共 {len(index_funds)} 家基金公司，通过被动指数产品持有002516。")
    lines.append("这些持仓源于指数纳入，非主动选股，信号意义在于反映股票市值/流动性变化。")
    lines.append("")
    lines.append("| 基金公司 | 产品数 | 合计持仓(万股) | 最新日期 |")
    lines.append("|----------|--------|---------------|---------|")
    for ip in index_funds:
        h = ip.get("latest_holdings", {})
        name = ip.get("short_name", "").replace("(指数)", "")
        lines.append(f"| {name} | {h.get('产品数量', '')} | {h.get('合计持仓_万股', '')} | {h.get('日期', '')} |")
    lines.append("")

    # Summary matrix (key institutions only)
    lines.append("---")
    lines.append("## 机构持仓矩阵（最新报告期）")
    lines.append("")
    lines.append("| 机构 | 类型 | 持仓(万股) | 占比 | 变动趋势 | 投资风格 |")
    lines.append("|------|------|-----------|------|---------|---------|")

    style_map = {
        "NOMURA-SUNRISE-001": ("1200", "0.82%", "零变动(4年+)", "战略持有"),
        "SHENJIELIANG-001": ("27399", "18.80%", "减持28%→启创一号", "实控人"),
        "QICHUANG-001": ("41183", "28.26%", "2025Q4新进", "产业投资"),
        "KUANGDA-HOLDING-001": ("7260", "0.50%", "长期稳定", "控股平台"),
        "KUANGDA-VC-001": ("4543", "3.12%", "未跟随减持", "系内PE"),
        "NORTHBOUND-001": ("1582", "1.09%", "波动增持", "交易型外资"),
        "CHANGZHOU-CAPITAL-001": ("2301", "1.56%", "长期稳定", "地方国资"),
        "ESOP-001": ("1168", "0.80%", "稳定", "员工持股"),
    }

    for p in deep:
        pid = p.get("profile_id", "")
        name = p.get("short_name", p.get("name", ""))
        inst_type = p.get("identity", {}).get("性质",
                    p.get("identity", {}).get("身份",
                    p.get("identity", {}).get("类型", "")))
        row = style_map.get(pid, ("", "", "", ""))
        lines.append(f"| {name} | {inst_type} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    # Total index fund holdings
    total_index = sum(
        float(ip.get("latest_holdings", {}).get("合计持仓_万股", 0))
        for ip in index_funds
    )
    lines.append(f"| 指数基金合计 | 被动配置 | {total_index:.0f} | ~{total_index/145700:.1f}% | 随指数调整 | 被动 |")
    lines.append("")

    # Key takeaways
    lines.append("## 关键发现")
    lines.append("")
    lines.append("### 1. 9月Level-2异动有明确公开催化：控制权变更")
    lines.append("")
    lines.append(
        "2025-09-06 公司披露筹划控制权变更复牌及股份转让协议，"
        "2025-09-10 披露权益变动报告书和财务顾问核查意见。"
        "这与 2025-09-08、09-10 的超级买入和 09-11 的集中卖出高度重合。"
        "所以9月大额行为首先应解释为控制权变更事件驱动，而不是单纯题材炒作。"
    )
    lines.append("")
    lines.append("### 2. 野村东方日出东方1号 = 长期锚定持仓，不能直接归因9/10扫货")
    lines.append("")
    lines.append(
        "野村东方日出东方1号是**单一资产管理计划（一对一专户）**，不是公募基金。"
        "公开披露只能从上市公司前十大/流通股东反推其持仓。"
        "它从2021Q4到2026Q1维持1200万股不变，说明它更像长期锚定持仓，"
        "而不是2025-09-10盘中超级买入的直接实名账户。"
        "上海纺织/东方国际/野村东方之间的关联闭环是强推断，后续要用公告、年报或工商资料核实。"
    )
    lines.append("")
    lines.append("### 3. 002516的股东结构正在重构")
    lines.append("")
    lines.append(
        "沈介良2025Q4转让4.1亿股（28%）给启创一号产业基金后，"
        "股东结构从'实控人一家独大'变为'实控人+产业基金+地方国资+外资'的多元化格局。"
        "这种结构更像为重大资本运作（如再融资、并购、分拆）做准备。"
    )
    lines.append("")
    lines.append("### 4. 可交易资金线索仍需分层")
    lines.append("")
    lines.append(
        "除北向资金外，多数持股>0.5%的机构偏静态。"
        "2025-09 的交易行为更可能来自事件驱动短线资金、北向/量化资金、"
        "或围绕控制权变更的资金博弈。野村这类静态持仓应作为股东结构背景，"
        "不要和盘中扫货账户混为一谈。"
    )
    lines.append("")
    lines.append("### 5. 后续追踪优先级")
    lines.append("")
    lines.append("1. **启创一号**（最高优先级）— 新进入，需确认是否为短期接盘方还是战略投资者")
    lines.append("2. **野村东方日出1号** — 关键观察：如果它开始减持，说明战略协同破裂或委托人策略转向")
    lines.append("3. **北向资金** — 唯一的高频交易信号源，可结合Level-2做盘中预警")
    lines.append("4. **旷达控股/旷达创业投资** — 实控人关联方，需追踪是否跟随意减持")
    lines.append("")

    # Write report
    report_path = output_dir / "institution_profiles.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def _dict_to_md(lines: list[str], d: dict, indent: int = 0):
    """Recursively render a dict to markdown bullet list."""
    prefix = "  " * indent
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            lines.append(f"{prefix}- **{k}**:")
            _dict_to_md(lines, v, indent + 1)
        elif isinstance(v, list):
            lines.append(f"{prefix}- **{k}**:")
            for item in v:
                if isinstance(item, dict):
                    factor = item.get("因素", "")
                    analysis = item.get("分析", "")
                    lines.append(f"{prefix}  - **{factor}** — {analysis}" if factor else f"{prefix}  - {analysis}")
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}- **{k}**: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("机构深度画像系统 — Institution Deep Profiling")
    print(f"标的: {STOCK} {STOCK_NAME}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] 加载数据...")
    evidence = load_evidence_data()
    prices = load_price_data()
    print(f"  公开证据 {len(evidence['public_evidence'])} 条")
    print(f"  持仓变化 {len(evidence['holder_changes'])} 条")
    print(f"  价格数据 {len(prices)} 天")

    # Extract institutions
    print("\n[2/5] 识别和分类机构...")
    institutions = extract_institutions(evidence["public_evidence"])
    print(f"  识别到 {len(institutions)} 条持仓记录")
    print(f"  唯一机构: {institutions['name'].nunique()} 个")

    # Show institution summary
    priority1 = institutions[institutions["priority"] == 1]
    priority2 = institutions[institutions["priority"] == 2]
    priority3 = institutions[institutions["priority"] == 3]

    print(f"\n  P1 (深度画像): {priority1['name'].nunique()} 个")
    for name in priority1["name"].unique():
        print(f"    - {name[:60]}")

    print(f"\n  P2 (标准画像): {priority2['name'].nunique()} 个")
    for name in priority2["name"].unique()[:10]:
        print(f"    - {name[:60]}")

    print(f"\n  P3 (统计追踪): {priority3['name'].nunique()} 个")

    # Build profiles
    print("\n[3/5] 构建机构画像...")
    all_profiles = []

    # L3: Deep profiles
    print("\n  --- L3 深度画像 ---")

    print("  [1] 野村东方日出东方1号...")
    nomura = profile_nomura_sunrise(institutions, prices)
    all_profiles.append(nomura)
    print(f"      持仓: {nomura.get('holdings_002516', {}).get('持仓数量_万股', 'N/A')}万股")
    print(f"      风格: {nomura.get('motivation', {}).get('投资风格', 'N/A')}")

    print("  [2] 沈介良(实控人)...")
    shen = profile_shen_jieliang(institutions, prices)
    all_profiles.append(shen)
    holdings = shen.get("holdings_002516", {})
    event = holdings.get("重大事件", {})
    if event:
        print(f"      最新持股: {holdings.get('最新持股_万股', 'N/A')}万股")
        print(f"      重大事件: {event.get('事件', 'N/A')}")

    print("  [3] 启创一号产业基金...")
    qc = profile_qichuang_one(institutions, prices)
    all_profiles.append(qc)
    print(f"      持仓: {qc.get('holdings_002516', {}).get('持股数量', 'N/A')}")

    # L2: Standard profiles
    print("\n  --- L2 标准画像 ---")

    print("  [4] 北向资金...")
    nb = profile_northbound(institutions, prices)
    all_profiles.append(nb)
    print(f"      趋势: {nb.get('holdings_002516', {}).get('趋势', 'N/A')}")

    print("  [5] 常州产投...")
    cz = profile_changzhou_capital(institutions, prices)
    all_profiles.append(cz)
    print(f"      持仓: {cz.get('holdings_002516', {}).get('持股数量', 'N/A')}")

    print("  [6] 旷达控股(实控人平台)...")
    kh = profile_kuangda_holding(institutions)
    all_profiles.append(kh)
    print(f"      持仓: {kh.get('holdings_002516', {}).get('持股数量', 'N/A')}")

    print("  [7] 旷达创投(系内PE)...")
    kv = profile_kuangda_venture(institutions)
    all_profiles.append(kv)
    print(f"      持仓: {kv.get('holdings_002516', {}).get('持股数量', 'N/A')}")

    print("  [8] 员工持股计划...")
    esop = profile_employees(institutions)
    if esop:
        all_profiles.append(esop)
        print(f"      持仓: {esop.get('holdings_002516', {}).get('持股数量', 'N/A')}")

    # L1: Index funds
    print("\n  --- L1 统计追踪 ---")
    print("  [9] 指数基金群...")
    index_profiles = profile_index_funds(institutions)
    all_profiles.extend(index_profiles)
    for ip in index_profiles:
        h = ip.get("latest_holdings", {})
        print(f"      {ip['short_name']}: {h.get('合计持仓_万股', 'N/A')}万股 ({h.get('产品数量', 'N/A')})")

    # Fetch supplementary info
    print("\n[4/5] 补充公开信息...")
    news = fetch_nomura_news()
    if news:
        print(f"  野村相关新闻: {len(news)}条")
        for n in news[:3]:
            print(f"    - {n['title'][:60]}")
    else:
        print("  未能获取野村相关新闻（网络或API限制）")

    # Generate report
    print("\n[5/5] 生成报告...")
    report_path = generate_report(all_profiles, STOCK, STOCK_NAME, OUTPUT_DIR)

    print(f"\n{'=' * 80}")
    print(f"报告已生成: {report_path}")
    print(f"画像数量: {len(all_profiles)} 个机构")
    print(f"  L3深度: {sum(1 for p in all_profiles if p.get('profile_level') == 'L3')}")
    print(f"  L2标准: {sum(1 for p in all_profiles if p.get('profile_level') == 'L2')}")
    print(f"  L1统计: {sum(1 for p in all_profiles if p.get('profile_level') == 'L1')}")
    print(f"{'=' * 80}")

    # Print key insight
    print("\n🔑 核心发现:")
    print("  2025年9月Level-2大额行为 ↔ 控制权变更公告链高度重合")
    print("  野村东方日出东方1号 = 长期锚定持仓线索，不能直接归因9/10扫货")
    print("  1200万股从2021年至今未变 → 偏静态，需继续核实上海纺织/东方国际关系")
    print("  真正在主动交易的资金线索 → 北向、量化、事件驱动短线资金")


if __name__ == "__main__":
    main()
