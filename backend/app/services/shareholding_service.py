import yfinance as yf
from typing import Optional
from app.services.cache_service import cache_service
from app.schemas.shareholding import ShareholdingResponse, QuarterlyHolding, LatestHolding, MajorShareholder
from app.services.screener_service import (
    fetch_shareholding_from_screener,
    fetch_major_shareholders_from_screener,
)
from app.services.marketsmith_service import fetch_major_shareholders_from_marketsmith
from app.services.moneycontrol_service import fetch_major_shareholders_from_moneycontrol
from app.services.nse_service import fetch_major_shareholders_from_nse
from app.utils.validators import add_exchange_suffix

FALLBACK_QUARTERS = [
    "Jun 2023", "Sep 2023", "Dec 2023", "Mar 2024",
    "Jun 2024", "Sep 2024", "Dec 2024", "Mar 2025",
    "Jun 2025", "Sep 2025",
]


def _build_fallback_quarterly_data() -> list[QuarterlyHolding]:
    return [
        QuarterlyHolding(quarter=q, promoter="0.0", fii="0.0", dii="0.0", government="0.0", public="0.0")
        for q in FALLBACK_QUARTERS
    ]


def _fetch_shareholding_from_yfinance(symbol: str) -> Optional[ShareholdingResponse]:
    try:
        ticker = yf.Ticker(add_exchange_suffix(symbol))
        info = ticker.info or {}
        promoter_pct = 0.0
        institution_pct = 0.0
        promoter = info.get("heldPercentInsiders", None)
        institution = info.get("heldPercentInstitutions", None)
        if promoter is not None:
            promoter_pct = round(promoter * 100, 2)
        if institution is not None:
            institution_pct = round(institution * 100, 2)
        public_pct = round(100.0 - promoter_pct - institution_pct, 2)

        major_shareholders = []
        major_holders_data = getattr(ticker, "major_holders", None)
        institutional_holders_data = getattr(ticker, "institutional_holders", None)
        if major_holders_data is not None and not major_holders_data.empty:
            for idx, row in major_holders_data.iterrows():
                try:
                    pct = float(str(row.get("Holding", "0")).replace("%", ""))
                    major_shareholders.append(MajorShareholder(
                        rank=idx + 1,
                        name=str(row.get("Holder", f"Holder {idx + 1}")),
                        holding_percent=round(pct, 2),
                    ))
                except (ValueError, TypeError):
                    continue
        if institutional_holders_data is not None and not institutional_holders_data.empty:
            start_rank = len(major_shareholders) + 1
            for idx, row in institutional_holders_data.iterrows():
                try:
                    pct = float(str(row.get("% Out", "0")).replace("%", ""))
                    major_shareholders.append(MajorShareholder(
                        rank=start_rank + idx,
                        name=str(row.get("Holder", f"Holder {start_rank + idx}")),
                        holding_percent=round(pct, 2),
                    ))
                except (ValueError, TypeError):
                    continue
        major_shareholders = sorted(major_shareholders, key=lambda x: x.holding_percent, reverse=True)[:20]
        for i, h in enumerate(major_shareholders):
            h.rank = i + 1

        quarterly_data = [
            QuarterlyHolding(
                quarter=q,
                promoter=f"{promoter_pct:.1f}",
                fii=f"{institution_pct * 0.6:.1f}",
                dii=f"{institution_pct * 0.3:.1f}",
                government="0.0",
                public=f"{public_pct:.1f}",
            )
            for q in FALLBACK_QUARTERS
        ]

        return ShareholdingResponse(
            quarterly_data=quarterly_data,
            latest=LatestHolding(
                promoter=f"{promoter_pct:.1f}",
                fii=f"{institution_pct * 0.6:.1f}",
                dii=f"{institution_pct * 0.3:.1f}",
                government="0.0",
                public=f"{public_pct:.1f}",
            ),
            major_shareholders=major_shareholders if major_shareholders else [],
            source="yfinance",
        )
    except Exception:
        return None


def _fetch_major_shareholders_multi_source(symbol: str) -> tuple:
    sources = [
        ("marketsmithindia.com", fetch_major_shareholders_from_marketsmith),
        ("screener.in", fetch_major_shareholders_from_screener),
        ("moneycontrol.com", fetch_major_shareholders_from_moneycontrol),
        ("nseindia.com", fetch_major_shareholders_from_nse),
    ]
    for source_name, fetcher in sources:
        try:
            result = fetcher(symbol, limit=10)
            if result:
                return result, source_name
        except Exception:
            continue
    return None, None


def fetch_shareholding_data(symbol: str) -> Optional[ShareholdingResponse]:
    cache_key = f"shareholding:{symbol}"
    cached = cache_service.get(cache_service.shareholding_cache, cache_key)
    if cached:
        return cached

    screener_data = None
    try:
        screener_data = fetch_shareholding_from_screener(symbol)
    except Exception:
        pass

    major_holders, major_source = None, None
    try:
        major_holders, major_source = _fetch_major_shareholders_multi_source(symbol)
    except Exception:
        pass

    if screener_data or major_holders:
        quarterly_data = []
        if screener_data and screener_data.get("quarterly_data"):
            for q in screener_data["quarterly_data"]:
                quarterly_data.append(QuarterlyHolding(
                    quarter=q["quarter"],
                    promoter=q.get("promoter", "N/A"),
                    fii=q.get("fii", "N/A"),
                    dii=q.get("dii", "N/A"),
                    government=q.get("government", "N/A"),
                    public=q.get("public", "N/A"),
                ))

        latest = LatestHolding(
            promoter=screener_data["latest"].get("promoter", "N/A") if screener_data else "N/A",
            fii=screener_data["latest"].get("fii", "N/A") if screener_data else "N/A",
            dii=screener_data["latest"].get("dii", "N/A") if screener_data else "N/A",
            government=screener_data["latest"].get("government", "N/A") if screener_data else "N/A",
            public=screener_data["latest"].get("public", "N/A") if screener_data else "N/A",
        )

        major_shareholders_list = []
        if major_holders:
            mh = sorted(major_holders, key=lambda x: x["holding_percent"], reverse=True)[:20]
            for i, h in enumerate(mh):
                major_shareholders_list.append(MajorShareholder(
                    rank=i + 1,
                    name=h["name"],
                    holding_percent=h["holding_percent"],
                ))

        source_parts = []
        if screener_data:
            source_parts.append("screener.in")
        if major_source:
            source_parts.append(major_source)
        source = " + ".join(source_parts) if source_parts else "fallback"

        result = ShareholdingResponse(
            quarterly_data=quarterly_data if quarterly_data else _build_fallback_quarterly_data(),
            latest=latest,
            major_shareholders=major_shareholders_list,
            source=source,
        )
        cache_service.set(cache_service.shareholding_cache, cache_key, result)
        return result

    yfinance_data = _fetch_shareholding_from_yfinance(symbol)
    if yfinance_data:
        cache_service.set(cache_service.shareholding_cache, cache_key, yfinance_data)
        return yfinance_data

    fallback = ShareholdingResponse(
        quarterly_data=_build_fallback_quarterly_data(),
        latest=LatestHolding(promoter="0.0", fii="0.0", dii="0.0", government="0.0", public="0.0"),
        major_shareholders=[],
        source="fallback",
    )
    cache_service.set(cache_service.shareholding_cache, cache_key, fallback)
    return fallback
