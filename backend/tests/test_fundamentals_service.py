import asyncio
import pytest
from app.services.fundamental_service import fundamentals_service
from app.routes.stock import get_stock_fundamentals, get_full_analysis
from app.schemas.analysis import FullAnalysisResponse
from app.schemas.fundamentals import FundamentalsResponse


def test_fundamental_categorization_banks():
    # Financial/Banking sectors have structured regulatory leverage
    cat = fundamentals_service._categorize_debt_to_equity(None, "Financial Services")
    assert cat["label"] == "N/A"
    assert cat["score"] > 80.0


def test_fundamental_categorization_industrials():
    cat_low = fundamentals_service._categorize_debt_to_equity(0.15, "Information Technology")
    assert cat_low["label"] == "Low"
    assert cat_low["score"] == 100.0

    cat_med = fundamentals_service._categorize_debt_to_equity(0.9, "Automobile")
    assert cat_med["label"] == "Medium"

    cat_high = fundamentals_service._categorize_debt_to_equity(2.5, "Telecom")
    assert cat_high["label"] == "High"


def test_score_calculation():
    # Scoring out of 100 with category breakdowns
    res = fundamentals_service._calculate_categories_and_score(
        pe=12.5,
        roe=0.18,
        roce=0.15,
        de=0.2,
        de_score=100.0,
        opm=0.22,
        rev_growth=0.15,
        profit_growth=0.18,
        sector="Information Technology",
    )
    assert 0 <= res["total"] <= 100
    assert res["rating"] in ["Strong Buy", "Buy"]
    assert len(res["categories"]) == 4
    weights_sum = sum(c["weight"] for c in res["categories"])
    assert weights_sum == 100.0


def test_get_stock_fundamentals_bankbaroda():
    resp = asyncio.run(get_stock_fundamentals("BANKBARODA"))
    assert isinstance(resp, FundamentalsResponse)
    f = resp.fundamentals
    assert f.market_cap is not None and f.market_cap > 0
    assert f.pe_ratio is not None and f.pe_ratio > 0
    assert f.roe is not None
    assert resp.score.total > 0
    assert len(resp.score.categories) == 4


def test_get_full_analysis_bankbaroda():
    raw_dict = asyncio.run(get_full_analysis("BANKBARODA"))
    validated = FullAnalysisResponse(**raw_dict)
    assert validated.fundamentals.market_cap is not None
    assert validated.fundamentals.roe is not None
    assert validated.score.total > 0
