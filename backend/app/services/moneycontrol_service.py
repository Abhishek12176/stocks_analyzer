import requests
import pandas as pd
from io import StringIO
from urllib.parse import quote_plus
from typing import Optional
from app.utils.scraper_utils import (
    HEADERS, extract_major_shareholders_from_tables, clean_text, clean_symbol,
)


def _find_moneycontrol_stock_code(symbol: str) -> Optional[str]:
    symbol = clean_symbol(symbol)
    search_url = (
        "https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php"
        f"?query={quote_plus(symbol)}&type=1&format=json"
    )
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        item = data[0]
        for key in ["sc_id", "id", "code"]:
            value = item.get(key)
            if value:
                return clean_text(value).upper()
        for key in ["link_src", "link", "url"]:
            link = item.get(key)
            if not link:
                continue
            parts = [p for p in clean_text(link).split("/") if p]
            if parts:
                return parts[-1].upper()
        return None
    except Exception:
        return None


def fetch_major_shareholders_from_moneycontrol(symbol: str, limit: int = 10) -> Optional[list]:
    symbol = clean_symbol(symbol)
    mc_code = _find_moneycontrol_stock_code(symbol)
    if not mc_code:
        return None
    urls = [
        f"https://m.moneycontrol.com/stock/{mc_code}/company-facts/shareholding-pattern",
        f"https://www.moneycontrol.com/company-facts/{mc_code}/shareholding-pattern",
    ]
    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            df = extract_major_shareholders_from_tables(tables, limit=limit)
            if df is not None:
                result = []
                for _, row in df.iterrows():
                    result.append({
                        "name": row["Shareholder"],
                        "holding_percent": row["Holding (%)"],
                    })
                return result
        except Exception:
            continue
    return None
