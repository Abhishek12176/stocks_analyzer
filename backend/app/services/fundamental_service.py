import logging
from datetime import timedelta

import pandas as pd
import yfinance as yf
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Indian SEBI filing deadline: 45 days from quarter-end for listed companies.
_QUARTERLY_FILING_LAG_DAYS = 45

class FundamentalsService:

    def _get_fallback(self, value, field: str, info: dict):
        if value is not None:
            return value
        fallback_map = {
            'marketCap': lambda: info.get('marketCap'),
            'trailingPE': lambda: info.get('trailingPE'),
            'trailingEps': lambda: info.get('trailingEps'),
            'returnOnEquity': lambda: info.get('returnOnEquity'),
            'returnOnCapitalEmployed': lambda: info.get('returnOnCapitalEmployed'),
            'debtToEquity': lambda: info.get('debtToEquity'),
            'operatingMargins': lambda: info.get('operatingMargins'),
            'revenueGrowth': lambda: info.get('revenueGrowth'),
            'earningsGrowth': lambda: info.get('earningsGrowth'),
        }
        fallback = fallback_map.get(field)
        if fallback:
            return fallback()
        return value

    def _calc_roce(self, ticker) -> Optional[float]:
        try:
            bs = ticker.balance_sheet
            inc = ticker.financials
            if bs is None or inc is None or bs.empty or inc.empty:
                return None

            bs_col = bs.iloc[:, 0]
            inc_col = inc.iloc[:, 0]

            total_assets = bs_col.get("Total Assets")
            current_liabilities = bs_col.get("Current Liabilities")
            total_equity = bs_col.get("Total Equity Gross Minority Interest")
            total_debt = bs_col.get("Total Debt")

            # Try EBIT first, fallback to Operating Income, then Net Income
            ebit = inc_col.get("EBIT") or inc_col.get("Operating Income")
            net_income = inc_col.get("Net Income")

            # Formula 1: EBIT / (Total Assets - Current Liabilities)
            if ebit is not None and total_assets is not None and current_liabilities is not None:
                capital_employed = total_assets - current_liabilities
                if capital_employed > 0:
                    return ebit / capital_employed

            # Formula 2: EBIT / (Total Equity + Total Debt) — works for financial companies
            if ebit is not None and total_equity is not None and total_debt is not None:
                capital_employed = total_equity + total_debt
                if capital_employed > 0:
                    return ebit / capital_employed

            # Formula 3: Net Income / (Total Equity + Total Debt) — for banks
            if net_income is not None and total_equity is not None and total_debt is not None:
                capital_employed = total_equity + total_debt
                if capital_employed > 0:
                    return net_income / capital_employed

            return None
        except Exception:
            return None

    def get_fundamentals(self, symbol: str, exchange: str, bse_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch fundamentals from yfinance and calculate scores
        """
        try:
            # Yahoo symbol convert
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{bse_code}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            # Extract raw data with fallbacks
            market_cap = info.get('marketCap')
            pe = info.get('trailingPE')
            eps = info.get('trailingEps')
            roe = info.get('returnOnEquity')
            roce = info.get('returnOnCapitalEmployed')
            de = info.get('debtToEquity')
            opm = info.get('operatingMargins')
            revenue_growth = info.get('revenueGrowth')
            profit_growth = info.get('earningsGrowth')
            sector = info.get('sector', 'Unknown')

            market_cap = self._get_fallback(market_cap, 'marketCap', info)
            pe = self._get_fallback(pe, 'trailingPE', info)
            eps = self._get_fallback(eps, 'trailingEps', info)
            roe = self._get_fallback(roe, 'returnOnEquity', info)
            de = self._get_fallback(de, 'debtToEquity', info)
            opm = self._get_fallback(opm, 'operatingMargins', info)
            revenue_growth = self._get_fallback(revenue_growth, 'revenueGrowth', info)
            profit_growth = self._get_fallback(profit_growth, 'earningsGrowth', info)

            # ROCE: try direct field first, then calculate from financial statements
            if roce is None:
                roce = self._calc_roce(ticker)

            # Calculate D/E category + score
            de_data = self._categorize_debt_to_equity(de, sector)

            # Calculate fundamental score out of 100
            fundamental_score = self._calculate_score({
                'pe': pe,
                'roe': roe,
                'de_score': de_data['score'],
                'opm': opm
            })

            # Rating based on score
            rating = self._get_rating(fundamental_score)

            return {
                "market_cap": market_cap,
                "pe_ratio": pe,
                "eps": eps,
                "roe": roe,
                "roce": roce,
                "debt_to_equity": de,
                "de_category": de_data['label'],
                "de_score": de_data['score'],
                "operating_margin": opm,
                "revenue_growth": revenue_growth,
                "profit_growth": profit_growth,
                "sector": sector,
                "fundamental_score": fundamental_score,
                "rating": rating,
                "summary": self._generate_summary(fundamental_score, de_data['label'], sector),
                "source_status": "real" if info else "partial_real_data"
            }

        except Exception as e:
            return {
                "source_status": "unavailable",
                "error": str(e),
                "fundamental_score": 0,
                "rating": "N/A"
            }

    def _categorize_debt_to_equity(self, de: Optional[float], sector: str) -> Dict[str, Any]:
        """
        Categorize Debt/Equity ratio based on sector
        Returns label + score out of 100
        """
        if de is None:
            return {"label": "N/A", "score": 0}

        # Banking/NBFC/Financial sector has high leverage normally
        financial_sectors = ["Banking", "NBFC", "Financial Services", "Insurance", "Capital Markets"]

        if any(fs in sector for fs in financial_sectors):
            # Bank/NBFC thresholds - higher D/E is normal
            if de <= 10:
                label = "Low"
                score = 100
            elif de <= 15:
                label = "Medium"
                score = max(0, 100 - ((de - 10) / 5) * 100) # Linear decay 10-15
            else:
                label = "High"
                score = max(0, 100 - (de - 10) * 5) # Decay faster after 15
        else:
            # Non-financial companies - your logic
            if de <= 0.5:
                label = "Low"
                score = 100
            elif de <= 1.5:
                label = "Medium"
                score = max(0, 100 - ((de - 0.5) / 1.0) * 100) # Linear decay 0.5 to 1.5
            else:
                label = "High"
                score = max(0, 100 - (de - 0.5) * 50) # Decay after 1.5

        return {"label": label, "score": round(score, 2)}

    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate overall fundamental score 0-100"""
        scores = []

        # D/E Score - 25 weight
        if metrics['de_score'] is not None:
            scores.append(metrics['de_score'] * 0.25)

        # ROE Score - 25 weight, >20% = 100
        if metrics.get('roe') is not None:
            roe_score = min(100, (metrics['roe'] * 100) * 5) # 20% ROE = 100
            scores.append(roe_score * 0.25)

        # PE Score - 25 weight, lower is better, 15 = 100
        if metrics.get('pe') is not None:
            pe_score = max(0, 100 - (metrics['pe'] - 15) * 2)
            scores.append(pe_score * 0.25)

        # OPM Score - 25 weight, >20% = 100
        if metrics.get('opm') is not None:
            opm_score = min(100, metrics['opm'] * 100 * 5)
            scores.append(opm_score * 0.25)

        return round(sum(scores) / len(scores), 2) if scores else 0

    def _get_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 80:
            return "Strong Buy"
        elif score >= 65:
            return "Buy"
        elif score >= 45:
            return "Hold"
        elif score >= 30:
            return "Sell"
        else:
            return "Strong Sell"

    def _generate_summary(self, score: float, de_label: str, sector: str) -> str:
        """Generate short summary based on available data"""
        if score >= 80:
            return f"Strong fundamentals with {de_label} debt level"
        elif score >= 65:
            return f"Good fundamentals, {de_label} debt level"
        else:
            return f"Weak fundamentals, {de_label} debt level. Research needed"

    def get_quarterly_fundamentals(
        self,
        symbol: str,
        exchange: str,
        quarter_end_prices: Optional[dict] = None,
        bse_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Historical quarterly PIT fundamental snapshots (for `enrich`/model input).

        Builds one snapshot per quarter-end from yfinance `quarterly_financials`
        and `quarterly_balance_sheet`. `available_at` uses the conservative SEBI
        filing-lag proxy `quarter_end + 45 days` (the last day results are legally
        required to be published) so no lookahead between quarter-end and release.

        `quarter_end_prices`: optional {quarter-end-date ISO: close price} map used
        to compute a trailing P/E for each quarter (price/EPS). When absent, P/E is
        NaN (left off the snapshot) — the remaining metrics still materialize.
        Returns an empty list on any failure (caller degrades gracefully).
        """
        try:
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{bse_code}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            qf = ticker.quarterly_financials
            qbs = ticker.quarterly_balance_sheet
            if qf is None or qf.empty or qbs is None or qbs.empty:
                return []
            # Drive on the financials columns (typically ~5 quarters); the
            # balance-sheet values fall back to the nearest prior balance-sheet
            # quarter via _cell's ffill lookup for rows missing a same-date entry.
            end_dates = sorted(set(qf.columns), reverse=True)
            if not end_dates:
                return []

            snapshots: List[Dict[str, Any]] = []
            # Year-over-year growth needs the same quarter one year back.
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

                pre_eps = _cell(qf, "Diluted EPS", prev_end)
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

                # Trailing P/E = quarter-end close EPS when a price map was given.
                pe = None
                if eps is not None and eps > 0 and quarter_end_prices:
                    price = quarter_end_prices.get(str(q_end.date()))
                    if price is None:
                        price = quarter_end_prices.get(q_end.date())
                    if price is not None and price > 0:
                        pe = price / eps

                de_data = self._categorize_debt_to_equity(de, "Unknown")
                fundamental_score = self._calculate_score(
                    {"pe": pe, "roe": roe, "de_score": de_data["score"], "opm": opm}
                ) if (pe is not None or roe is not None or opm is not None or de is not None) else None

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
        except Exception as exc:  # noqa: BLE001 — caller degrades gracefully
            logger.warning("quarterly fundamentals unavailable for %s: %s", symbol, exc)
            return []


def _cell(df: pd.DataFrame, row: str, col: pd.Timestamp):
    """Read one cell from a yfinance quarterly frame (rows = metrics, cols = dates).

    Falls back to the nearest value at-or-before a later column when the exact
    `col` is absent (yfinance balance sheets carry fewer quarters than income
    statements; the columns arrive in descending order, so this lookup runs on
    an ascending-sorted copy).
    """
    try:
        if df is None or df.empty or row not in df.index:
            return None
        cols = pd.Index([pd.Timestamp(c).normalize() for c in df.columns if pd.notna(c)])
        if len(cols) == 0:
            return None
        asc = cols.sort_values()  # ascending-copy index
        target = pd.Timestamp(col).normalize()
        # position of largest value <= target
        pos = asc.searchsorted(target, side="right") - 1
        if pos < 0:
            return None
        chosen = asc[pos]
        val = df.loc[row, chosen]
        return float(val) if val is not None and pd.notna(val) else None
    except Exception:
        return None


fundamentals_service = FundamentalsService()
