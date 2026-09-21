"""Unit tests for the upgraded chat intent/symbol layer (no network)."""

import asyncio

import pytest

import app.services.chat_service as chat_service
from app.config import settings
from app.services.chat_service import (
    _analyst_report,
    _compact_fundamentals,
    _concept_kb_reply,
    _financial_disclaimer,
    _general_reply,
    _resolve_symbols,
    _ticker_candidates,
    extract_symbols,
    filter_predictions,
    is_smalltalk,
    is_stock_query,
    parse_intent,
    process_chat,
)


class TestExtractSymbols:
    def test_bare_symbol(self):
        assert extract_symbols("RELIANCE ka signal?") == ["RELIANCE"]

    def test_lowercase_symbol(self):
        assert extract_symbols("tcs ka price kya hai") == ["TCS"]

    def test_dot_ns_suffix(self):
        assert extract_symbols("check infosys.ns") == ["INFY"]

    def test_company_name_map(self):
        assert "RELIANCE" in extract_symbols("reliance industries ke share")
        assert "HDFCBANK" in extract_symbols("hdfc bank")
        assert "SBIN" in extract_symbols("state bank of india")
        assert "BHARTIARTL" in extract_symbols("airtel")

    def test_no_symbols(self):
        assert extract_symbols("aaj mausam kaisa hai") == []

    def test_dedupe(self):
        assert extract_symbols("RELIANCE vs reliance industries") == ["RELIANCE"]

    def test_ambiguous_bare_name_not_matched(self):
        # "tata" alone is intentionally not mapped (could be many companies).
        assert extract_symbols("tata ke baare me batao") == []
        assert extract_symbols("adani") == []


class TestParseIntent:
    def test_under(self):
        assert parse_intent("stocks under Rs 500")["maxPrice"] == 500.0

    def test_hinglish_under(self):
        assert parse_intent("500 se kam rate wale stocks")["maxPrice"] == 500.0

    def test_hinglish_above(self):
        assert parse_intent("1000 se zyada wale")["minPrice"] == 1000.0

    def test_duration_not_price(self):
        # Bug fix: "20 din ka" must NOT be read as maxPrice=20.
        intent = parse_intent("20 din ka BUY signal do")
        assert intent["maxPrice"] is None
        assert intent["minPrice"] is None
        assert intent["action"] == "buy"

    def test_percent_not_price(self):
        intent = parse_intent("5% growth wale")
        assert intent["maxPrice"] is None

    def test_bare_number_is_price(self):
        intent = parse_intent("200 wale stocks do")
        assert intent["maxPrice"] == 200.0

    def test_price_with_rupaye_word(self):
        intent = parse_intent("100 rupaye se niche wale stocks do")
        assert intent["maxPrice"] == 100.0

    def test_price_with_dash_slash(self):
        intent = parse_intent("100/- se niche ke stock do")
        assert intent["maxPrice"] == 100.0

    def test_price_with_ke_under(self):
        intent = parse_intent("100 ke under stocks")
        assert intent["maxPrice"] == 100.0

    def test_min_price_with_rupaye(self):
        intent = parse_intent("2000 rupaye se zyada wale")
        assert intent["minPrice"] == 2000.0

    def test_strong_bullish_action(self):
        intent = parse_intent("strong bullish wale stock do")
        assert intent["action"] == "strong_buy"

    def test_strong_bearish_action(self):
        intent = parse_intent("strong bearish stocks dikhao")
        assert intent["action"] == "strong_sell"

    def test_bullish_maps_to_buy(self):
        intent = parse_intent("bullish signal wale stocks")
        assert intent["action"] == "buy"

    def test_bearish_maps_to_sell(self):
        intent = parse_intent("bearish stocks")
        assert intent["action"] == "sell"

    def test_bullish_days(self):
        intent = parse_intent("5 din se bullish stock do")
        assert intent["bullishDays"] == 5
        assert intent["maxPrice"] is None
        assert intent["minPrice"] is None

    def test_bullish_days_upar(self):
        assert parse_intent("10 din se upar wale stocks")["bullishDays"] == 10

    def test_bullish_days_badh_raha(self):
        assert parse_intent("3 din se badh raha stock")["bullishDays"] == 3

    def test_bullish_days_capped(self):
        assert parse_intent("99 din se bullish")["bullishDays"] == 60

    def test_duration_signal_not_bullish(self):
        # "din" alone must not set bullishDays or a price — only matched with a
        # bullish/upar/badh-raha keyword.
        intent = parse_intent("5 din ka BUY signal do")
        assert intent["bullishDays"] is None
        assert intent["maxPrice"] is None
        assert intent["action"] == "buy"


class TestRouting:
    def test_smalltalk(self):
        assert is_smalltalk("hi")
        assert is_smalltalk("thank you")
        assert not is_smalltalk("hi RELIANCE kaise hai")
        assert not is_smalltalk("hello, 500 se kam stocks batao")

    def test_stock_query_by_symbol(self):
        assert is_stock_query("RELIANCE kaisi hai?")
        assert is_stock_query("hdfc bank ka analysis")

    def test_stock_query_by_keyword(self):
        assert is_stock_query("top buy signals")
        assert is_stock_query("stocks under 500")

    def test_educational_not_query(self):
        assert not is_stock_query("what is RSI")
        assert not is_stock_query("RSI kya hota hai")
        assert not is_stock_query("stock market kaise kaam karta hai")

    def test_bullish_days_is_stock_query(self):
        assert is_stock_query("10 din se upar wale stocks")
        assert is_stock_query("3 din se badh raha stock")
        assert not is_smalltalk("5 din se bullish stock do")


class TestCompactFundamentals:
    def test_none(self):
        assert _compact_fundamentals(None) is None

    def test_pct_conversion(self):
        out = _compact_fundamentals(
            {
                "sector": "Energy",
                "pe_ratio": 24.0,
                "roe": 0.098,
                "operating_margin": 0.142,
                "revenue_growth": -0.006,
                "profit_growth": None,
                "fundamental_score": 62,
                "rating": "Outperform",
            }
        )
        assert out["roe"] == 9.8
        assert out["opm"] == 14.2
        assert out["revenueGrowth"] == -0.6
        assert out["pe"] == 24.0
        assert out["score"] == 62
        assert out["profitGrowth"] is None
        assert out["pe"] == 24.0


def _bullish_ctx():
    return {
        "quote": {"price": 100.0, "changePercent": 1.5},
        "indicators": {"rsi": 55.0, "macd": 1.2, "macdSignal": 0.5, "sma20": 95.0, "sma50": 90.0},
        "fundamentals": {"pe": 12.0, "roe": 18.0, "opm": 25.0, "revenueGrowth": 10.0, "profitGrowth": 8.0, "debtEquity": 0.4},
        "sentiment": 0.3,
        "news": [],
    }


class TestAnalystReport:
    def test_bullish(self):
        report = _analyst_report(_bullish_ctx(), {"latest": {"probability": 0.72, "signal": "BUY"}})
        assert report["action"] in ("BUY", "STRONG BUY")
        assert report["score"] >= 60

    def test_bearish(self):
        ctx = _bullish_ctx()
        ctx["indicators"] = {"rsi": 78.0, "macd": -1.2, "macdSignal": 0.5, "sma20": 90.0, "sma50": 95.0}
        ctx["sentiment"] = -0.4
        ctx["fundamentals"] = {"pe": 80.0, "roe": -5.0, "opm": 2.0, "revenueGrowth": -3.0, "profitGrowth": -10.0, "debtEquity": 3.0}
        report = _analyst_report(ctx, {"latest": {"probability": 0.28, "signal": "SELL"}})
        assert report["action"] in ("SELL", "STRONG SELL")
        assert report["score"] <= 45

    def test_empty_reasonable(self):
        report = _analyst_report({"quote": {}, "indicators": {}, "fundamentals": {}, "sentiment": 0.0, "news": []}, None)
        assert 0 <= report["score"] <= 100
        assert report["action"] == "HOLD"

    def test_score_clamped(self):
        r = _analyst_report(_bullish_ctx(), {"latest": {"probability": 0.99, "signal": "BUY"}})
        assert r["score"] <= 100


class TestConceptKb:
    def test_rsi(self):
        assert "RSI" in (_concept_kb_reply("what is RSI") or "")

    def test_pe(self):
        assert "PE" in (_concept_kb_reply("pe ratio kya hota hai") or "")

    def test_hinglish_variant(self):
        assert "RSI" in (_concept_kb_reply("RSI kya hai?") or "")

    def test_unmatched(self):
        assert _concept_kb_reply("aaj mausam kaisa hai") is None


class TestDeepDiveEmptyData:
    def test_empty_ctx_is_honest(self):
        from app.services.chat_service import _fallback_deep_dive
        out = _fallback_deep_dive("ZOMATO", {"quote": {}, "indicators": {}, "fundamentals": {}, "sentiment": 0.0, "news": []}, None, False)
        assert "ZOMATO" in out
        assert "fetch" in out.lower()
        assert "Verdict" not in out

    def test_ctx_with_price_renders_report(self):
        from app.services.chat_service import _fallback_deep_dive
        ctx = {"quote": {"price": 150.0, "changePercent": 1.5}, "indicators": {"rsi": 55.0, "macd": 1.0, "macdSignal": 0.5, "sma20": 140.0, "sma50": 135.0}, "fundamentals": {"pe": 20.0, "roe": 12.0}, "sentiment": 0.1, "news": []}
        out = _fallback_deep_dive("ZOMATO", ctx, None, False)
        assert "Verdict" in out
        assert "₹150.00" in out


class TestNewCompanyAliases:
    def test_suzlon(self):
        assert extract_symbols("suzlon energy ka signal") == ["SUZLON"]

    def test_idbi(self):
        assert extract_symbols("idbi ke share") == ["IDBI"]

    def test_yes_bank(self):
        assert extract_symbols("yes bank ka price") == ["YESBANK"]

    def test_zomato_lowercase(self):
        assert extract_symbols("zomato ka kya haal hai") == ["ZOMATO"]


class TestTickerCandidates:
    def test_uppercase_ticker(self):
        assert _ticker_candidates("ZOMATO ka kya haal hai") == ["ZOMATO"]

    def test_caps_ticker_with_suffix(self):
        assert _ticker_candidates("check IRCTC.NS") == ["IRCTC"]

    def test_caps_ignored_for_common_words(self):
        assert _ticker_candidates("TOP STOCKS please") == []
        assert _ticker_candidates("BEST BUY signals") == []

    def test_lowercase_words_not_tickers(self):
        assert _ticker_candidates("please tell me") == []

    def test_resolve_known_and_unknown(self):
        assert _resolve_symbols("RELIANCE aur ZOMATO") == ["RELIANCE", "ZOMATO"]


class TestFilterPredictions:
    def test_strong_buy_uses_category(self):
        stocks = [
            {"symbol": "A", "signal": "BUY", "confidence": 90, "currentPrice": 100, "categories": ["strong-buy", "buy"]},
            {"symbol": "B", "signal": "BUY", "confidence": 72, "currentPrice": 100, "categories": ["buy"]},
        ]
        out = filter_predictions(stocks, {"maxPrice": None, "minPrice": None, "action": "strong_buy", "top": 10})
        assert [s["symbol"] for s in out] == ["A"]

    def test_strong_sell_uses_category(self):
        stocks = [
            {"symbol": "C", "signal": "SELL", "confidence": 90, "currentPrice": 100, "categories": ["strong-sell"]},
            {"symbol": "D", "signal": "SELL", "confidence": 80, "currentPrice": 100, "categories": ["sell"]},
        ]
        out = filter_predictions(stocks, {"maxPrice": None, "minPrice": None, "action": "strong_sell", "top": 10})
        assert [s["symbol"] for s in out] == ["C"]

    def test_price_and_action_combined(self):
        stocks = [
            {"symbol": "E", "signal": "BUY", "confidence": 80, "currentPrice": 120, "categories": ["buy"]},
            {"symbol": "F", "signal": "BUY", "confidence": 80, "currentPrice": 80, "categories": ["buy"]},
        ]
        out = filter_predictions(stocks, {"maxPrice": 100, "minPrice": None, "action": "buy", "top": 10})
        assert [s["symbol"] for s in out] == ["F"]

    def test_price_missing_skipped(self):
        stocks = [{"symbol": "G", "signal": "BUY", "confidence": 80, "currentPrice": None, "categories": ["buy"]}]
        assert filter_predictions(stocks, {"maxPrice": None, "minPrice": None, "action": None, "top": 10}) == []

    def test_bullish_days_uses_positive_return(self):
        stocks = [
            {"symbol": "A", "signal": "BUY", "confidence": 80, "currentPrice": 10, "categories": ["buy"], "ret5d": 2.5},
            {"symbol": "B", "signal": "BUY", "confidence": 80, "currentPrice": 20, "categories": ["buy"], "ret5d": -1.0},
            {"symbol": "C", "signal": "BUY", "confidence": 80, "currentPrice": 30, "categories": ["buy"], "ret5d": 0.0},
            {"symbol": "D", "signal": "BUY", "confidence": 80, "currentPrice": 40, "categories": ["buy"], "ret15d": 5.0},
        ]
        out = filter_predictions(stocks, {"maxPrice": None, "minPrice": None, "action": None, "top": 10, "bullishDays": 5})
        assert [s["symbol"] for s in out] == ["A"]

    def test_bullish_days_falls_back_to_streak(self):
        stocks = [
            {"symbol": "A", "signal": "BUY", "confidence": 80, "currentPrice": 10, "categories": ["buy"], "upDaysConsecutive": 30},
            {"symbol": "B", "signal": "BUY", "confidence": 80, "currentPrice": 20, "categories": ["buy"], "upDaysConsecutive": 20},
        ]
        out = filter_predictions(stocks, {"maxPrice": None, "minPrice": None, "action": None, "top": 10, "bullishDays": 30})
        assert [s["symbol"] for s in out] == ["A"]

    def test_bullish_days_no_ret_and_no_streak_excluded(self):
        stocks = [
            {"symbol": "A", "signal": "BUY", "confidence": 80, "currentPrice": 10, "categories": ["buy"], "upDaysConsecutive": 60},
            {"symbol": "B", "signal": "BUY", "confidence": 80, "currentPrice": 20, "categories": ["buy"], "upDaysConsecutive": None},
        ]
        out = filter_predictions(stocks, {"maxPrice": None, "minPrice": None, "action": None, "top": 10, "bullishDays": 60})
        assert [s["symbol"] for s in out] == ["A"]


class TestFinancialDisclaimer:
    def test_disclaimer_added_to_stock_reply(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(
            chat_service,
            "load_predictions",
            lambda: [
                {"symbol": "AAA", "signal": "BUY", "confidence": 80, "currentPrice": 120,
                 "categories": ["buy"], "predictionLabel": "BUY", "prediction20d": "P(up) 68%"},
                {"symbol": "BBB", "signal": "BUY", "confidence": 60, "currentPrice": 90,
                 "categories": ["buy"], "predictionLabel": "BUY", "prediction20d": "P(up) 60%"},
            ],
        )
        result = asyncio.run(process_chat("buy signal wale stocks", []))
        assert result["source"] == "existing-model"
        assert "financial advice" in result["reply"]
        assert _financial_disclaimer(False) in result["reply"] or _financial_disclaimer(True) in result["reply"]

    def test_disclaimer_added_to_general_stock_chat(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "")
        reply = asyncio.run(_general_reply("tell me about stock market investing strategies", []))
        assert "financial advice" in reply

    def test_disclaimer_added_to_concept_kb(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "")
        reply = asyncio.run(_general_reply("what is RSI", []))
        assert "RSI" in reply
        assert "financial advice" in reply

    def test_no_disclaimer_on_who_made_you(self):
        reply = asyncio.run(_general_reply("kisne banaya tumhe", []))
        assert "financial advice" not in reply

    def test_no_disclaimer_on_out_of_scope(self):
        reply = asyncio.run(_general_reply("aaj mausam kaisa hai", []))
        assert "financial advice" not in reply

    def test_no_disclaimer_on_smalltalk(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "")
        result = asyncio.run(process_chat("hello", []))
        assert "financial advice" not in result["reply"]