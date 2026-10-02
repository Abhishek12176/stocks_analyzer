import logging
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from cachetools import TTLCache

from app.utils.scraper_utils import fetch_html
from app.utils.validators import clean_symbol

logger = logging.getLogger(__name__)

# Indian SEBI filing deadline: 45 days from quarter-end for listed companies.
_QUARTERLY_FILING_LAG_DAYS = 45


class FundamentalsService:
    def __init__(self):
        # Cache results in-memory for 1 hour to ensure ultra-fast subsequent responses
        self._cache: TTLCache = TTLCache(maxsize=300, ttl=3600)

    def _parse_num(self, val: Any) -> Optional[float]:
        if val is None or val == "" or val == "-":
            return None
        try:
            cleaned = str(val).replace(",", "").replace("%", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _fetch_screener_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real, comprehensive fundamental data for Indian stocks directly from Screener.in.
        Works seamlessly and reliably without cloud rate-limits.
        """
        clean = clean_symbol(symbol)
        url = f"https://www.screener.in/company/{clean}/"
        html = fetch_html(url)
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")

            # 1. Top Key Ratios
            ratios: Dict[str, str] = {}
            for li in soup.find_all("li", class_="flex"):
                name_el = li.find("span", class_="name")
                num_el = li.find("span", class_="number")
                if name_el and num_el:
                    ratios[name_el.text.strip()] = num_el.text.strip().replace(",", "")

            mcap_cr = self._parse_num(ratios.get("Market Cap"))
            market_cap = float(mcap_cr * 10_000_000) if mcap_cr is not None else None
            pe = self._parse_num(ratios.get("Stock P/E"))
            book_value = self._parse_num(ratios.get("Book Value"))
            div_yield = self._parse_num(ratios.get("Dividend Yield"))

            roce_raw = self._parse_num(ratios.get("ROCE"))
            roce = float(roce_raw / 100.0) if roce_raw is not None else None

            roe_raw = self._parse_num(ratios.get("ROE"))
            roe = float(roe_raw / 100.0) if roe_raw is not None else None

            # 2. Sector Identification
            sector = "Unknown"
            peer_sec = soup.find("section", id="peers")
            if peer_sec:
                for a in peer_sec.find_all("a"):
                    if "/sector/" in a.get("href", ""):
                        sector = a.text.strip()
                        break
            if sector == "Unknown":
                tag = soup.find("a", href=re.compile(r"/market/"))
                if tag:
                    sector = tag.text.strip()

            is_financial = any(
                fs in sector.lower()
                for fs in ["bank", "financial", "nbfc", "insurance", "capital market"]
            )

            # 3. Profit & Loss table (EPS, OPM, Growth)
            eps = None
            opm = None
            sales_growth = None
            profit_growth = None

            pl = soup.find("section", id="profit-loss")
            if pl:
                for tr in pl.find_all("tr"):
                    cells = [
                        c.text.strip().replace("\xa0", "").replace("\n", "").replace(",", "").replace("%", "")
                        for c in tr.find_all(["th", "td"])
                    ]
                    if not cells:
                        continue
                    rn = cells[0].lower()
                    if "eps" in rn:
                        vals = [self._parse_num(x) for x in cells[1:] if self._parse_num(x) is not None]
                        if vals:
                            eps = vals[-1]
                    elif "opm" in rn or "financing margin" in rn:
                        vals = [self._parse_num(x) for x in cells[1:] if self._parse_num(x) is not None]
                        if vals:
                            opm = vals[-1] / 100.0

                for box in pl.find_all("table", class_="ranges-table"):
                    txt = box.text
                    if "Sales Growth" in txt:
                        for tr in box.find_all("tr"):
                            row_txt = tr.text
                            if any(k in row_txt for k in ["TTM", "1 Year", "3 Years"]):
                                m = re.search(r"(-?\d+)%", row_txt)
                                if m and sales_growth is None:
                                    sales_growth = float(m.group(1)) / 100.0
                    if "Profit Growth" in txt:
                        for tr in box.find_all("tr"):
                            row_txt = tr.text
                            if any(k in row_txt for k in ["TTM", "1 Year", "3 Years"]):
                                m = re.search(r"(-?\d+)%", row_txt)
                                if m and profit_growth is None:
                                    profit_growth = float(m.group(1)) / 100.0

            # 4. Debt to Equity (Borrowings / Equity Capital + Reserves)
            de = None
            if not is_financial:
                bs = soup.find("section", id="balance-sheet")
                if bs:
                    borrowings = 0.0
                    equity = 0.0
                    for tr in bs.find_all("tr"):
                        cells = [
                            c.text.strip().replace("\xa0", "").replace("\n", "").replace(",", "")
                            for c in tr.find_all(["th", "td"])
                        ]
                        if not cells:
                            continue
                        rn = cells[0].lower()
                        if "borrowing" in rn:
                            vals = [self._parse_num(x) for x in cells[1:] if self._parse_num(x) is not None]
                            if vals:
                                borrowings = vals[-1]
                        elif "equity capital" in rn or "share capital" in rn:
                            vals = [self._parse_num(x) for x in cells[1:] if self._parse_num(x) is not None]
                            if vals:
                                equity += vals[-1]
                        elif "reserves" in rn:
                            vals = [self._parse_num(x) for x in cells[1:] if self._parse_num(x) is not None]
                            if vals:
                                equity += vals[-1]
                    if equity > 0:
                        de = round(borrowings / equity, 2)

            # PB Ratio
            pb_ratio = None
            if pe is not None and roe is not None and roe > 0:
                pb_ratio = round(pe * roe, 2)
            elif market_cap is not None and book_value is not None and book_value > 0 and eps:
                # Estimate P/B = Current Price / Book Value
                curr_price = self._parse_num(ratios.get("Current Price"))
                if curr_price:
                    pb_ratio = round(curr_price / book_value, 2)

            return {
                "market_cap": market_cap,
                "pe_ratio": pe,
                "eps": eps,
                "book_value": book_value,
                "dividend_yield": div_yield,
                "roe": roe,
                "roce": roce,
                "debt_to_equity": de,
                "operating_margin": opm,
                "revenue_growth": sales_growth,
                "profit_growth": profit_growth,
                "pb_ratio": pb_ratio,
                "sector": sector,
            }
        except Exception as e:
            logger.warning("Error parsing Screener fundamentals for %s: %s", symbol, e)
            return None

    def _fetch_yfinance_fundamentals(
        self, symbol: str, exchange: str, bse_code: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fallback to Yahoo Finance if needed."""
        try:
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{bse_code}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info or {}
            if not info or not info.get("marketCap"):
                return None

            return {
                "market_cap": self._parse_num(info.get("marketCap")),
                "pe_ratio": self._parse_num(info.get("trailingPE")),
                "eps": self._parse_num(info.get("trailingEps")),
                "book_value": self._parse_num(info.get("bookValue")),
                "dividend_yield": self._parse_num(info.get("dividendYield")),
                "roe": self._parse_num(info.get("returnOnEquity")),
                "roce": self._parse_num(info.get("returnOnCapitalEmployed")),
                "debt_to_equity": self._parse_num(info.get("debtToEquity")),
                "operating_margin": self._parse_num(info.get("operatingMargins")),
                "revenue_growth": self._parse_num(info.get("revenueGrowth")),
                "profit_growth": self._parse_num(info.get("earningsGrowth")),
                "sector": info.get("sector", "Unknown"),
            }
        except Exception as e:
            logger.debug("yfinance fundamentals unavailable for %s: %s", symbol, e)
            return None

    def _categorize_debt_to_equity(self, de: Optional[float], sector: str) -> Dict[str, Any]:
        """
        Categorize Debt/Equity ratio accurately based on sector.
        Returns label and score out of 100.
        """
        is_financial = any(
            fs in sector.lower()
            for fs in ["bank", "financial", "nbfc", "insurance", "capital market"]
        )

        if is_financial:
            # Banking/NBFC leverage is structural (deposits/RBI capital adequacy)
            return {"label": "N/A", "score": 85.0}

        if de is None:
            return {"label": "Low", "score": 80.0}

        if de <= 0.3:
            label = "Low"
            score = 100.0
        elif de <= 0.8:
            label = "Low"
            score = 85.0
        elif de <= 1.5:
            label = "Medium"
            score = max(50.0, round(85.0 - ((de - 0.8) / 0.7) * 35.0, 1))
        else:
            label = "High"
            score = max(20.0, round(50.0 - (de - 1.5) * 20.0, 1))

        return {"label": label, "score": float(score)}

    def _calculate_categories_and_score(
        self,
        pe: Optional[float],
        roe: Optional[float],
        roce: Optional[float],
        de: Optional[float],
        de_score: float,
        opm: Optional[float],
        rev_growth: Optional[float],
        profit_growth: Optional[float],
        sector: str,
    ) -> Dict[str, Any]:
        """
        Calculate category scores (out of 25 each) and overall 0-100 fundamental score.
        """
        # 1. Valuation Score (Weight 25)
        if pe is not None and pe > 0:
            if pe <= 10:
                val_score = 24.0
            elif pe <= 18:
                val_score = 21.0
            elif pe <= 28:
                val_score = 17.0
            elif pe <= 40:
                val_score = 12.0
            elif pe <= 60:
                val_score = 7.0
            else:
                val_score = 4.0
        else:
            val_score = 14.0

        # 2. Profitability Score (Weight 25: ROE 15 + ROCE 10)
        roe_pts = 8.0
        if roe is not None:
            r_pct = roe * 100
            if r_pct >= 20:
                roe_pts = 15.0
            elif r_pct >= 14:
                roe_pts = 12.0
            elif r_pct >= 10:
                roe_pts = 9.0
            elif r_pct > 0:
                roe_pts = 6.0
            else:
                roe_pts = 2.0

        roce_pts = 5.0
        if roce is not None:
            rc_pct = roce * 100
            if rc_pct >= 20:
                roce_pts = 10.0
            elif rc_pct >= 12:
                roce_pts = 8.0
            elif rc_pct >= 6:
                roce_pts = 6.0
            elif rc_pct > 0:
                roce_pts = 4.0
            else:
                roce_pts = 1.0

        prof_score = round(roe_pts + roce_pts, 1)

        # 3. Financial Health (Weight 25)
        health_score = round((de_score / 100.0) * 25.0, 1)

        # 4. Growth (Weight 25: Sales Growth 12.5 + Profit Growth 12.5)
        rg_pts = 7.0
        if rev_growth is not None:
            rg_pct = rev_growth * 100
            if rg_pct >= 15:
                rg_pts = 12.5
            elif rg_pct >= 8:
                rg_pts = 10.0
            elif rg_pct >= 0:
                rg_pts = 7.0
            else:
                rg_pts = 3.5

        pg_pts = 7.0
        if profit_growth is not None:
            pg_pct = profit_growth * 100
            if pg_pct >= 15:
                pg_pts = 12.5
            elif pg_pct >= 8:
                pg_pts = 10.0
            elif pg_pct >= 0:
                pg_pts = 7.0
            else:
                pg_pts = 3.5

        growth_score = round(rg_pts + pg_pts, 1)

        total = round(val_score + prof_score + health_score + growth_score, 1)
        total = max(0.0, min(100.0, total))

        if total >= 75:
            rating = "Strong Buy"
        elif total >= 65:
            rating = "Buy"
        elif total >= 48:
            rating = "Hold"
        elif total >= 35:
            rating = "Sell"
        else:
            rating = "Strong Sell"

        score_color = "green" if total >= 65 else "gold" if total >= 45 else "red"

        categories = [
            {
                "name": "Valuation",
                "score": float(val_score),
                "weight": 25.0,
                "metrics": [
                    {"name": "P/E Ratio", "value": float(pe) if pe is not None else None, "max_score": 25.0}
                ],
            },
            {
                "name": "Profitability",
                "score": float(prof_score),
                "weight": 25.0,
                "metrics": [
                    {"name": "ROE", "value": round(float(roe) * 100, 2) if roe is not None else None, "max_score": 15.0},
                    {"name": "ROCE", "value": round(float(roce) * 100, 2) if roce is not None else None, "max_score": 10.0},
                ],
            },
            {
                "name": "Financial Health",
                "score": float(health_score),
                "weight": 25.0,
                "metrics": [
                    {"name": "Debt to Equity", "value": float(de) if de is not None else None, "max_score": 25.0}
                ],
            },
            {
                "name": "Growth",
                "score": float(growth_score),
                "weight": 25.0,
                "metrics": [
                    {"name": "Revenue Growth", "value": round(float(rev_growth) * 100, 2) if rev_growth is not None else None, "max_score": 12.5},
                    {"name": "Profit Growth", "value": round(float(profit_growth) * 100, 2) if profit_growth is not None else None, "max_score": 12.5},
                ],
            },
        ]

        return {
            "total": float(total),
            "rating": rating,
            "score_color": score_color,
            "categories": categories,
        }

    def _generate_summary(self, score: float, de_label: str, sector: str) -> str:
        """Generate concise summary for fundamentals."""
        if score >= 75:
            return f"Robust financial foundation with {de_label.lower()} debt risk and strong profitability in {sector}."
        elif score >= 60:
            return f"Healthy fundamentals with sustainable returns and stable capital structure."
        elif score >= 45:
            return f"Moderate fundamentals. Potential growth with fair valuation and monitorable debt."
        else:
            return f"Caution advised. Stretched valuation or compressed margins observed."

    def get_fundamentals(self, symbol: str, exchange: str = "NSE", bse_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified fundamentals retriever with multi-source fallback (Screener.in + yfinance)
        and caching.
        """
        clean = clean_symbol(symbol)
        cache_key = f"{clean}_{exchange}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Primary: Screener.in (fast, accurate for Indian equities, unblocked on Render)
        data = self._fetch_screener_fundamentals(clean)

        # 2. Secondary fallback/supplement: yfinance
        if not data or data.get("market_cap") is None:
            yf_data = self._fetch_yfinance_fundamentals(clean, exchange, bse_code)
            if yf_data:
                if not data:
                    data = yf_data
                else:
                    for k, v in yf_data.items():
                        if data.get(k) is None:
                            data[k] = v

        if not data:
            return {
                "source_status": "unavailable",
                "error": "No fundamental data found",
                "fundamental_score": 0.0,
                "rating": "N/A",
                "categories": [],
            }

        # Format sector
        sector = data.get("sector") or "Unknown"

        # Categorize D/E
        de = data.get("debt_to_equity")
        de_data = self._categorize_debt_to_equity(de, sector)

        # Scoring
        score_data = self._calculate_categories_and_score(
            pe=data.get("pe_ratio"),
            roe=data.get("roe"),
            roce=data.get("roce"),
            de=de,
            de_score=de_data["score"],
            opm=data.get("operating_margin"),
            rev_growth=data.get("revenue_growth"),
            profit_growth=data.get("profit_growth"),
            sector=sector,
        )

        result = {
            "market_cap": float(data["market_cap"]) if data.get("market_cap") is not None else None,
            "pe_ratio": float(data["pe_ratio"]) if data.get("pe_ratio") is not None else None,
            "eps": float(data["eps"]) if data.get("eps") is not None else None,
            "book_value": float(data["book_value"]) if data.get("book_value") is not None else None,
            "dividend_yield": float(data["dividend_yield"]) if data.get("dividend_yield") is not None else None,
            "roe": float(data["roe"]) if data.get("roe") is not None else None,
            "roce": float(data["roce"]) if data.get("roce") is not None else None,
            "debt_to_equity": float(de) if de is not None else None,
            "de_category": de_data["label"],
            "de_score": float(de_data["score"]),
            "operating_margin": float(data["operating_margin"]) if data.get("operating_margin") is not None else None,
            "revenue_growth": float(data["revenue_growth"]) if data.get("revenue_growth") is not None else None,
            "profit_growth": float(data["profit_growth"]) if data.get("profit_growth") is not None else None,
            "pb_ratio": float(data["pb_ratio"]) if data.get("pb_ratio") is not None else None,
            "sector": sector,
            "fundamental_score": score_data["total"],
            "rating": score_data["rating"],
            "rating_color": score_data["score_color"],
            "categories": score_data["categories"],
            "summary": self._generate_summary(score_data["total"], de_data["label"], sector),
            "source_status": "real",
        }

        self._cache[cache_key] = result
        return result

    def get_quarterly_fundamentals(
        self,
        symbol: str,
        exchange: str,
        quarter_end_prices: Optional[dict] = None,
        bse_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Historical quarterly PIT fundamental snapshots (for `enrich`/model input)."""
        try:
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{bse_code}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            qf = ticker.quarterly_financials
            qbs = ticker.quarterly_balance_sheet
            if qf is None or qf.empty or qbs is None or qbs.empty:
                return []
            end_dates = sorted(set(qf.columns), reverse=True)
            if not end_dates:
                return []

            snapshots: List[Dict[str, Any]] = []
            for end_date in end_dates:
                q_end = pd.Timestamp(end_date).normalize()
                prev_end = q_end - pd.DateOffset(months=12)
                eps = _cell(qf, "Diluted EPS", q_end)
                operating_income = _cell(qf, "Operating Income", q_end)
                total_revenue = _cell(qf, "Total Revenue", q_end)
                net_income = _cell(qf, "Net Income", q_end)
                total_equity = _cell(qbs, "Total Equity Gross Minority Interest", q_end)
                total_debt = _cell(qbs, "Total Debt", q_end)
                current_liabilities = _cell(qbs, "Current Liabilities", q_end)
                total_assets = _cell(qbs, "Total Assets", q_end)

                pre_revenue = _cell(qf, "Total Revenue", prev_end)
                pre_net_income = _cell(qf, "Net Income", prev_end)

                roce = None
                if operating_income is not None and total_assets is not None and current_liabilities is not None:
                    capital = total_assets - current_liabilities
                    if capital > 0:
                        roce = operating_income / capital
                elif operating_income is not None and total_equity is not None and total_debt is not None:
                    capital = total_equity + total_debt
                    if capital > 0:
                        roce = operating_income / capital

                roe = None
                if net_income is not None and total_equity:
                    roe = net_income / total_equity

                de = None
                if total_debt is not None and total_equity:
                    de = total_debt / total_equity

                opm = None
                if operating_income is not None and total_revenue:
                    opm = operating_income / total_revenue

                def _growth(cur, prev):
                    if cur is None or prev is None or prev == 0:
                        return None
                    return cur / prev - 1

                rev_growth = _growth(total_revenue, pre_revenue)
                profit_growth = _growth(net_income, pre_net_income)

                pe = None
                if eps is not None and eps > 0 and quarter_end_prices:
                    price = quarter_end_prices.get(str(q_end.date()))
                    if price is None:
                        price = quarter_end_prices.get(q_end.date())
                    if price is not None and price > 0:
                        pe = price / eps

                de_data = self._categorize_debt_to_equity(de, "Unknown")
                score_res = self._calculate_categories_and_score(
                    pe=pe,
                    roe=roe,
                    roce=roce,
                    de=de,
                    de_score=de_data["score"],
                    opm=opm,
                    rev_growth=rev_growth,
                    profit_growth=profit_growth,
                    sector="Unknown",
                )
                fundamental_score = score_res["total"]

                snapshots.append({
                    "available_at": (q_end + timedelta(days=_QUARTERLY_FILING_LAG_DAYS)).date().isoformat(),
                    "pe_ratio": pe,
                    "eps": eps,
                    "roe": roe,
                    "roce": roce,
                    "debt_to_equity": de,
                    "operating_margin": opm,
                    "revenue_growth": rev_growth,
                    "profit_growth": profit_growth,
                    "fundamental_score": fundamental_score,
                })
            return snapshots
        except Exception as exc:
            logger.warning("quarterly fundamentals unavailable for %s: %s", symbol, exc)
            return []


def _cell(df: pd.DataFrame, row: str, col: pd.Timestamp):
    """Read one cell from a yfinance quarterly frame."""
    try:
        if df is None or df.empty or row not in df.index:
            return None
        cols = pd.Index([pd.Timestamp(c).normalize() for c in df.columns if pd.notna(c)])
        if len(cols) == 0:
            return None
        asc = cols.sort_values()
        target = pd.Timestamp(col).normalize()
        pos = asc.searchsorted(target, side="right") - 1
        if pos < 0:
            return None
        chosen = asc[pos]
        val = df.loc[row, chosen]
        return float(val) if val is not None and pd.notna(val) else None
    except Exception:
        return None


fundamentals_service = FundamentalsService()
