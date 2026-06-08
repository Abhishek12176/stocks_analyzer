import yfinance as yf
from typing import Dict, Any, Optional

class FundamentalsService:

    def get_fundamentals(self, symbol: str, exchange: str, bse_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch fundamentals from yfinance and calculate scores
        """
        try:
            # Yahoo symbol convert
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{bse_code}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            # Extract raw data
            market_cap = info.get('marketCap')
            pe = info.get('trailingPE')
            eps = info.get('trailingEps')
            roe = info.get('returnOnEquity')
            roce = info.get('returnOnCapitalEmployed') # May be null
            de = info.get('debtToEquity')
            opm = info.get('operatingMargins')
            sector = info.get('sector', 'Unknown')
            revenue_growth = info.get('revenueGrowth')
            profit_growth = info.get('earningsGrowth')

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
                "sector": sector,
                "revenue_growth": revenue_growth,
                "profit_growth": profit_growth,
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
        if metrics['roe']:
            roe_score = min(100, (metrics['roe'] * 100) * 5) # 20% ROE = 100
            scores.append(roe_score * 0.25)

        # PE Score - 25 weight, lower is better, 15 = 100
        if metrics['pe']:
            pe_score = max(0, 100 - (metrics['pe'] - 15) * 2)
            scores.append(pe_score * 0.25)

        # OPM Score - 25 weight, >20% = 100
        if metrics['opm']:
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

fundamentals_service = FundamentalsService()
