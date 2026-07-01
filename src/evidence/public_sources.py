"""Public-market evidence sources for stock behavior chains.

All fetchers are best-effort. A network/API failure should not block the local
Level-2 evidence chain.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


EASTMONEY_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


@dataclass(frozen=True)
class PublicFetchResult:
    source: str
    ok: bool
    data: pd.DataFrame
    error: str = ""


def _json_get(url: str, timeout: int = 15) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": EASTMONEY_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def _json_post(url: str, data: dict[str, str], timeout: int = 15) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "User-Agent": EASTMONEY_UA,
            "Referer": "http://www.cninfo.com.cn/",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def fetch_lhb_summary(stock_code: str, page_size: int = 100) -> PublicFetchResult:
    """Fetch Eastmoney daily billboard summaries for one stock."""
    filter_expr = urllib.parse.quote(f'(SECURITY_CODE="{stock_code}")', safe="()=")
    url = (
        "https://datacenter-web.eastmoney.com/api/data/v1/get?"
        "sortColumns=TRADE_DATE&sortTypes=-1"
        f"&pageSize={page_size}&pageNumber=1"
        "&reportName=RPT_BILLBOARD_DAILYDETAILS&columns=ALL"
        f"&filter={filter_expr}"
    )
    try:
        payload = _json_get(url)
        rows = (payload.get("result") or {}).get("data") or []
        data = pd.DataFrame(rows)
        if not data.empty and "TRADE_DATE" in data.columns:
            data["date"] = pd.to_datetime(data["TRADE_DATE"]).dt.strftime("%Y%m%d")
        return PublicFetchResult("eastmoney_lhb_summary", True, data)
    except Exception as exc:  # noqa: BLE001 - best-effort public fetch
        return PublicFetchResult("eastmoney_lhb_summary", False, pd.DataFrame(), str(exc))


def fetch_eastmoney_announcements(
    secucode: str,
    start_date: str,
    end_date: str,
    page_size: int = 100,
) -> PublicFetchResult:
    """Fetch Eastmoney announcement list. secucode example: 002516.SZ."""
    url = (
        "https://np-anotice-stock.eastmoney.com/api/security/ann?"
        f"sr=-1&page_size={page_size}&page_index=1&ann_type=A"
        f"&client_source=web&stock_list={urllib.parse.quote(secucode)}"
        f"&f_node=0&s_node=0&begin_time={start_date}&end_time={end_date}"
    )
    try:
        payload = _json_get(url)
        rows = ((payload.get("data") or {}).get("list")) or []
        data = pd.DataFrame(rows)
        if not data.empty:
            date_col = "notice_date" if "notice_date" in data.columns else "display_time"
            if date_col in data.columns:
                data["date"] = pd.to_datetime(data[date_col], errors="coerce").dt.strftime("%Y%m%d")
        return PublicFetchResult("eastmoney_announcements", True, data)
    except Exception as exc:  # noqa: BLE001
        return PublicFetchResult("eastmoney_announcements", False, pd.DataFrame(), str(exc))


def fetch_cninfo_announcements(
    stock_code: str,
    start_date: str,
    end_date: str,
    page_size: int = 100,
) -> PublicFetchResult:
    """Fetch CNInfo announcements; kept as a second opinion to Eastmoney."""
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    payload = {
        "pageNum": "1",
        "pageSize": str(page_size),
        "column": "szse" if stock_code.startswith(("0", "3")) else "sse",
        "tabName": "fulltext",
        "plate": "sz" if stock_code.startswith(("0", "3")) else "sh",
        "stock": stock_code,
        "searchkey": "",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{start_date}~{end_date}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    try:
        response = _json_post(url, payload)
        rows = response.get("announcements") or []
        data = pd.DataFrame(rows)
        if not data.empty and "announcementTime" in data.columns:
            data["date"] = pd.to_datetime(data["announcementTime"], unit="ms").dt.strftime("%Y%m%d")
        return PublicFetchResult("cninfo_announcements", True, data)
    except Exception as exc:  # noqa: BLE001
        return PublicFetchResult("cninfo_announcements", False, pd.DataFrame(), str(exc))


def fetch_eastmoney_news(stock_code: str, page_size: int = 100) -> PublicFetchResult:
    """Fetch Eastmoney stock news search results."""
    callback = "jQuery35101792940631092459_1764599530165"
    inner_param = {
        "uid": "",
        "keyword": stock_code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": page_size,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    params = urllib.parse.urlencode(
        {
            "cb": callback,
            "param": json.dumps(inner_param, ensure_ascii=False),
            "_": "1764599530176",
        }
    )
    url = f"https://search-api-web.eastmoney.com/search/jsonp?{params}"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": EASTMONEY_UA,
                "Referer": f"https://so.eastmoney.com/news/s?keyword={stock_code}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        if text.startswith(callback):
            text = text[len(callback) + 1 : -1]
        payload = json.loads(text)
        rows = ((payload.get("result") or {}).get("cmsArticleWebOld")) or []
        data = pd.DataFrame(rows)
        if not data.empty:
            data["date"] = pd.to_datetime(data.get("date"), errors="coerce").dt.strftime("%Y%m%d")
            data["title_clean"] = data.get("title", "").astype(str).map(_clean_html)
            data["content_clean"] = data.get("content", "").astype(str).map(_clean_html)
            if "code" in data.columns:
                data["url"] = "http://finance.eastmoney.com/a/" + data["code"].astype(str) + ".html"
        return PublicFetchResult("eastmoney_news", True, data)
    except Exception as exc:  # noqa: BLE001
        return PublicFetchResult("eastmoney_news", False, pd.DataFrame(), str(exc))


def fetch_akshare_public_tables(stock_code: str) -> list[PublicFetchResult]:
    """Fetch AkShare tables that can provide delayed institution evidence."""
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        return [PublicFetchResult("akshare_import", False, pd.DataFrame(), str(exc))]

    specs = [
        ("sina_fund_holders", lambda: ak.stock_fund_stock_holder(symbol=stock_code)),
        ("sina_main_holders", lambda: ak.stock_main_stock_holder(stock=stock_code)),
        ("sina_circulate_holders", lambda: ak.stock_circulate_stock_holder(symbol=stock_code)),
        ("ths_shareholder_change", lambda: ak.stock_shareholder_change_ths(symbol=stock_code)),
        ("eastmoney_research_reports", lambda: ak.stock_research_report_em(symbol=stock_code)),
        (
            "eastmoney_individual_notices",
            lambda: ak.stock_individual_notice_report(
                security=stock_code,
                symbol="全部",
                begin_date="2025-01-01",
                end_date="2026-12-31",
            ),
        ),
        (
            "eastmoney_block_trades",
            lambda: _filter_block_trades(
                ak.stock_dzjy_mrmx(symbol="A股", start_date="20250101", end_date="20261231"),
                stock_code,
            ),
        ),
    ]
    results: list[PublicFetchResult] = []
    for source, func in specs:
        try:
            data = func()
            if data is None:
                data = pd.DataFrame()
            results.append(PublicFetchResult(source, True, data))
        except Exception as exc:  # noqa: BLE001
            results.append(PublicFetchResult(source, False, pd.DataFrame(), str(exc)))
    return results


def fetch_public_evidence(
    stock_code: str,
    secucode: str,
    event_dates: list[str],
    lookaround_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and normalize public evidence around event dates."""
    if event_dates:
        min_date = datetime.strptime(min(event_dates), "%Y%m%d") - timedelta(days=lookaround_days)
        max_date = datetime.strptime(max(event_dates), "%Y%m%d") + timedelta(days=lookaround_days)
        start = min_date.strftime("%Y-%m-%d")
        end = max_date.strftime("%Y-%m-%d")
    else:
        start = "2025-01-01"
        end = "2025-12-31"

    results = [
        fetch_lhb_summary(stock_code),
        fetch_eastmoney_announcements(secucode, start, end),
        fetch_cninfo_announcements(stock_code, start, end),
        fetch_eastmoney_news(stock_code),
    ]
    results.extend(fetch_akshare_public_tables(stock_code))

    evidence_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for result in results:
        status_rows.append(
            {
                "source": result.source,
                "ok": result.ok,
                "rows": len(result.data),
                "error": result.error,
            }
        )
        if result.data.empty:
            continue
        evidence_rows.extend(_normalize_source(result.source, result.data))

    evidence = pd.DataFrame(evidence_rows)
    if not evidence.empty:
        evidence = evidence.sort_values(["date", "source", "title"]).reset_index(drop=True)
    return evidence, pd.DataFrame(status_rows)


def _normalize_source(source: str, data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if source == "eastmoney_lhb_summary":
        for _, row in data.iterrows():
            rows.append(
                {
                    "date": row.get("date", ""),
                    "source": source,
                    "evidence_type": "龙虎榜",
                    "title": row.get("EXPLANATION", ""),
                    "buy_amt": row.get("TOTAL_BUY", row.get("BILLBOARD_BUY_AMT", "")),
                    "sell_amt": row.get("TOTAL_SELL", row.get("BILLBOARD_SELL_AMT", "")),
                    "net_amt": row.get("TOTAL_NET", row.get("BILLBOARD_NET_AMT", "")),
                    "url": "",
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source in {"eastmoney_announcements", "cninfo_announcements"}:
        for _, row in data.iterrows():
            title = (
                row.get("title")
                or row.get("announcementTitle")
                or row.get("art_code")
                or row.get("columns")
                or ""
            )
            url = row.get("attach_url") or row.get("adjunctUrl") or row.get("url") or ""
            rows.append(
                {
                    "date": row.get("date", ""),
                    "source": source,
                    "evidence_type": "公告",
                    "title": title,
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": "",
                    "url": url,
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source == "eastmoney_news":
        for _, row in data.iterrows():
            code = row.get("code", "")
            url = row.get("url", f"http://finance.eastmoney.com/a/{code}.html" if code else "")
            rows.append(
                {
                    "date": row.get("date", ""),
                    "source": source,
                    "evidence_type": "新闻",
                    "title": row.get("title_clean", row.get("title", "")),
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": "",
                    "url": url,
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source == "sina_fund_holders":
        for _, row in data.iterrows():
            date = _date_to_yyyymmdd(row.get("截止日期"))
            rows.append(
                {
                    "date": date,
                    "source": source,
                    "evidence_type": "基金持仓",
                    "title": (
                        f"{row.get('基金名称', '')} 持仓{row.get('持仓数量', '')}股 "
                        f"占流通{row.get('占流通股比例', '')}%"
                    ),
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": row.get("持股市值", ""),
                    "url": "https://vip.stock.finance.sina.com.cn/",
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source in {"sina_main_holders", "sina_circulate_holders"}:
        date_col = "截至日期" if "截至日期" in data.columns else "截止日期"
        ratio_col = "持股比例" if "持股比例" in data.columns else "占流通股比例"
        for _, row in data.iterrows():
            date = _date_to_yyyymmdd(row.get(date_col))
            rows.append(
                {
                    "date": date,
                    "source": source,
                    "evidence_type": "股东持仓",
                    "title": (
                        f"{row.get('股东名称', '')} 持股{row.get('持股数量', '')} "
                        f"占比{row.get(ratio_col, '')}%"
                    ),
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": row.get("持股数量", ""),
                    "url": "https://vip.stock.finance.sina.com.cn/",
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source == "ths_shareholder_change":
        for _, row in data.iterrows():
            date = _date_to_yyyymmdd(row.get("公告日期"))
            rows.append(
                {
                    "date": date,
                    "source": source,
                    "evidence_type": "股东变动",
                    "title": (
                        f"{row.get('变动股东', '')} {row.get('变动数量', '')} "
                        f"{row.get('变动途径', '')} 均价{row.get('交易均价', '')}"
                    ),
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": "",
                    "url": "https://basic.10jqka.com.cn/",
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source == "eastmoney_research_reports":
        for _, row in data.iterrows():
            rows.append(
                {
                    "date": _date_to_yyyymmdd(row.get("日期")),
                    "source": source,
                    "evidence_type": "研报",
                    "title": f"{row.get('机构', '')} {row.get('东财评级', '')} {row.get('报告名称', '')}",
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": "",
                    "url": row.get("报告PDF链接", ""),
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source == "eastmoney_individual_notices":
        for _, row in data.iterrows():
            rows.append(
                {
                    "date": _date_to_yyyymmdd(row.get("公告日期")),
                    "source": source,
                    "evidence_type": "公告",
                    "title": f"{row.get('公告类型', '')} {row.get('公告标题', '')}",
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": "",
                    "url": row.get("网址", ""),
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    elif source == "eastmoney_block_trades":
        for _, row in data.iterrows():
            rows.append(
                {
                    "date": _date_to_yyyymmdd(row.get("交易日期")),
                    "source": source,
                    "evidence_type": "大宗交易",
                    "title": (
                        f"{row.get('证券简称', '')} 成交{row.get('成交量', '')}股 "
                        f"成交额{row.get('成交额', '')} 买方:{row.get('买方营业部', '')} "
                        f"卖方:{row.get('卖方营业部', '')}"
                    ),
                    "buy_amt": "",
                    "sell_amt": "",
                    "net_amt": row.get("成交额", ""),
                    "url": "https://data.eastmoney.com/dzjy/dzjy_mrmx.html",
                    "raw": json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str),
                }
            )
    return rows


def _clean_html(value: str) -> str:
    value = re.sub(r"</?em>", "", str(value))
    value = value.replace("\\u3000", "").replace("\u3000", "")
    value = value.replace("\r\n", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def _date_to_yyyymmdd(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return date.strftime("%Y%m%d")


def _filter_block_trades(data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if "证券代码" not in data.columns:
        return pd.DataFrame()
    code = data["证券代码"].astype(str).str.zfill(6)
    return data[code.eq(stock_code)].copy()


def match_public_events(
    behavior: pd.DataFrame,
    public_evidence: pd.DataFrame,
    lookaround_days: int = 3,
) -> pd.DataFrame:
    """Attach public events within +/- lookaround days to behavior rows."""
    frame = behavior.copy()
    if public_evidence.empty:
        frame["public_event_count"] = 0
        frame["public_events"] = ""
        frame["has_lhb"] = False
        frame["has_announcement"] = False
        return frame

    evidence = public_evidence.copy()
    evidence["date_ts"] = pd.to_datetime(evidence["date"], errors="coerce")
    frame["date_ts"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")

    counts: list[int] = []
    labels: list[str] = []
    has_lhb: list[bool] = []
    has_ann: list[bool] = []
    for _, row in frame.iterrows():
        start = row["date_ts"] - pd.Timedelta(days=lookaround_days)
        end = row["date_ts"] + pd.Timedelta(days=lookaround_days)
        matched = evidence[(evidence["date_ts"] >= start) & (evidence["date_ts"] <= end)]
        counts.append(len(matched))
        has_lhb.append((matched["evidence_type"] == "龙虎榜").any())
        has_ann.append((matched["evidence_type"] == "公告").any())
        items = []
        for _, event in matched.head(5).iterrows():
            items.append(f"{event['date']} {event['evidence_type']} {event['title']}")
        labels.append(" | ".join(items))

    frame["public_event_count"] = counts
    frame["public_events"] = labels
    frame["has_lhb"] = has_lhb
    frame["has_announcement"] = has_ann
    return frame.drop(columns=["date_ts"])


def compute_holder_changes(public_evidence: pd.DataFrame) -> pd.DataFrame:
    """Compute report-over-report holder/fund position changes."""
    if public_evidence.empty or "raw" not in public_evidence.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    holder_sources = {"sina_fund_holders", "sina_main_holders", "sina_circulate_holders"}
    for _, row in public_evidence[public_evidence["source"].isin(holder_sources)].iterrows():
        try:
            raw = json.loads(row.get("raw", "{}"))
        except Exception:  # noqa: BLE001
            continue

        source = row["source"]
        if source == "sina_fund_holders":
            name = raw.get("基金名称")
            date = raw.get("截止日期")
            shares = raw.get("持仓数量")
            ratio = raw.get("占流通股比例")
            holder_type = "基金"
        elif source == "sina_main_holders":
            name = raw.get("股东名称")
            date = raw.get("截至日期")
            shares = raw.get("持股数量")
            ratio = raw.get("持股比例")
            holder_type = "主要股东"
        else:
            name = raw.get("股东名称")
            date = raw.get("截止日期")
            shares = raw.get("持股数量")
            ratio = raw.get("占流通股比例")
            holder_type = "流通股东"

        rows.append(
            {
                "source": source,
                "holder_type": holder_type,
                "holder_name": name,
                "date": pd.to_datetime(date, errors="coerce"),
                "shares": pd.to_numeric(shares, errors="coerce"),
                "ratio": pd.to_numeric(ratio, errors="coerce"),
            }
        )

    if not rows:
        return pd.DataFrame()

    changes = pd.DataFrame(rows).dropna(subset=["holder_name", "date"])
    changes = changes.sort_values(["source", "holder_name", "date"])
    grouped = changes.groupby(["source", "holder_name"], dropna=False)
    changes["prev_date"] = grouped["date"].shift(1)
    changes["prev_shares"] = grouped["shares"].shift(1)
    changes["prev_ratio"] = grouped["ratio"].shift(1)
    changes["share_delta"] = changes["shares"] - changes["prev_shares"]
    changes["ratio_delta"] = changes["ratio"] - changes["prev_ratio"]
    changes["date"] = changes["date"].dt.strftime("%Y%m%d")
    changes["prev_date"] = changes["prev_date"].dt.strftime("%Y%m%d")
    changes = changes.sort_values(["date", "source", "holder_name"]).reset_index(drop=True)
    return changes
