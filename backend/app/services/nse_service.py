import requests
import pandas as pd
from io import StringIO
from typing import Optional
from app.utils.scraper_utils import HEADERS, extract_major_shareholders_from_tables, clean_symbol


def fetch_major_shareholders_from_nse(symbol: str, limit: int = 10) -> Optional[list]:
    symbol = clean_symbol(symbol)
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(
            f"https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
            f"?symbol={symbol}&tabIndex=equity",
            timeout=15,
        )
        api_url = f"https://www.nseindia.com/api/corporate-share-holdings?index=equities&symbol={symbol}"
        response = session.get(api_url, timeout=20)
        response.raise_for_status()
        filings = response.json()
        if not filings:
            return None
        latest = filings[0]
        xbrl_url = latest.get("xbrl")
        if not xbrl_url:
            return None
        possible_urls = [
            xbrl_url,
            xbrl_url.replace("/corporate/xbrl/", "/corporate/ixbrl/").replace(
                "_WEB.xml", "_iXBRL_WEB.html"
            ),
        ]
        for filing_url in possible_urls:
            try:
                html_response = session.get(filing_url, timeout=25)
                html_response.raise_for_status()
                tables = pd.read_html(StringIO(html_response.text))
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
    except Exception:
        return None
