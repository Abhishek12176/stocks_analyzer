import numpy as np
import pandas as pd
from typing import Optional
from app.utils.scraper_utils import parse_percent, finalize_major_shareholders, clean_symbol


def fetch_major_shareholders_from_marketsmith(symbol: str, limit: int = 10) -> Optional[list]:
    symbol = clean_symbol(symbol).lower()
    url = f"https://marketsmithindia.com/mstool/eval/{symbol}/evaluation.jsp#/"
    rows_data = []
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(4000)
            page.wait_for_selector(".industryGroup.tableBorderCss", timeout=15000)
            sections = page.locator(".industryGroup.tableBorderCss")
            total = sections.count()
            for i in range(total):
                section = sections.nth(i)
                section_text = section.inner_text()
                if "Major Shareholders" in section_text:
                    rows = section.locator("table tr")
                    row_count = rows.count()
                    for r in range(row_count):
                        cells = rows.nth(r).locator("th, td")
                        cell_count = cells.count()
                        if cell_count >= 2:
                            investor_name = cells.nth(0).inner_text().strip()
                            holding_text = cells.nth(1).inner_text().strip()
                            if investor_name.lower() == "investor name":
                                continue
                            holding = parse_percent(holding_text)
                            if investor_name and not pd.isna(holding) and holding > 0 and holding <= 100:
                                rows_data.append({
                                    "Shareholder": investor_name,
                                    "Holding (%)": round(holding, 2),
                                })
                    break
            browser.close()
    except Exception:
        return None
    if not rows_data:
        return None
    df = finalize_major_shareholders(rows_data, limit=limit)
    if df is None:
        return None
    result = []
    for _, row in df.iterrows():
        result.append({
            "name": row["Shareholder"],
            "holding_percent": row["Holding (%)"],
        })
    return result
