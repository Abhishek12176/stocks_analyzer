import re
import requests as req
import pandas as pd
from io import StringIO
from typing import Optional
from app.utils.scraper_utils import (
    fetch_html, find_shareholding_table, build_quarterly_data,
    extract_major_shareholders_from_tables, clean_symbol,
    HEADERS,
)


def fetch_shareholding_from_screener(symbol: str) -> Optional[dict]:
    symbol = clean_symbol(symbol)
    url = f"https://www.screener.in/company/{symbol}/"
    html = fetch_html(url)
    if not html:
        return None
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return None
    shareholding_table = find_shareholding_table(tables)
    if shareholding_table is None:
        return None
    quarterly_df = build_quarterly_data(shareholding_table)
    if quarterly_df.empty:
        return None
    row = quarterly_df.iloc[0]
    latest = {
        "promoter": row.get("Promoter (%)", "N/A"),
        "fii": row.get("FII/FPI (%)", "N/A"),
        "dii": row.get("DII (%)", "N/A"),
        "government": row.get("Government (%)", "N/A"),
        "public": row.get("Public (%)", "N/A"),
    }
    quarterly_data = []
    for _, r in quarterly_df.iterrows():
        quarterly_data.append({
            "quarter": r.get("Quarter", ""),
            "promoter": r.get("Promoter (%)", "N/A"),
            "fii": r.get("FII/FPI (%)", "N/A"),
            "dii": r.get("DII (%)", "N/A"),
            "government": r.get("Government (%)", "N/A"),
            "public": r.get("Public (%)", "N/A"),
        })
    return {"quarterly_data": quarterly_data, "latest": latest}


def _extract_company_id(html: str) -> Optional[str]:
    for m in re.finditer(r'data-url="/company/chat/(\d+)/"', html):
        return m.group(1)
    for m in re.finditer(r'data-url="/notebook/(\d+)/"', html):
        return m.group(1)
    return None


def _parse_latest_quarter(investor_data: dict) -> Optional[float]:
    quarters = [k for k in investor_data if k != "setAttributes"]
    if not quarters:
        return None
    latest = quarters[-1]
    val = investor_data[latest]
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_major_shareholders_from_screener(symbol: str, limit: int = 10) -> Optional[list]:
    symbol = clean_symbol(symbol)
    url = f"https://www.screener.in/company/{symbol}/"
    html = fetch_html(url)
    if not html:
        return None
    company_id = _extract_company_id(html)
    if not company_id:
        return None
    session = req.Session()
    session.headers.update(HEADERS)
    rows = []
    for cls in ["promoters", "foreign_institutions", "domestic_institutions", "government", "public"]:
        api_url = f"https://www.screener.in/api/3/{company_id}/investors/{cls}/quarterly/"
        try:
            resp = session.get(api_url, timeout=15, headers={"Accept": "application/json", **HEADERS})
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not isinstance(data, dict):
                continue
            for investor_name, quarter_data in data.items():
                if investor_name == "setAttributes":
                    continue
                holding = _parse_latest_quarter(quarter_data)
                if holding is None or holding <= 0 or holding > 100:
                    continue
                rows.append({
                    "name": investor_name,
                    "holding_percent": round(holding, 2),
                })
        except Exception:
            continue
    if not rows:
        return None
    rows = sorted(rows, key=lambda x: x["holding_percent"], reverse=True)[:limit]
    return rows
