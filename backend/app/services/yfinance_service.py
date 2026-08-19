import yfinance as yf
import pandas as pd
from app.services.cache_service import cache_service
from app.services.indicator_service import add_indicators, get_latest_indicators
from app.utils.validators import add_exchange_suffix
from app.utils.exceptions import StockNotFoundError, ServiceUnavailableError


def fetch_price_data(symbol: str, period: str = "1y") -> dict:
    """Fetch OHLCV data with technical indicators for a symbol."""
    cache_key = f"price_{symbol}_{period}"

    cached = cache_service.get(cache_service.price_cache, cache_key)
    if cached:
        return cached

    try:
        ticker = add_exchange_suffix(symbol)
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            raise StockNotFoundError(symbol)

        # Clean multi-level columns if present
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].iloc[:, 0]
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data.dropna(inplace=True)

        # Add indicators
        data = add_indicators(data)
        latest_indicators = get_latest_indicators(data)

        # Get quote info (handle transient failures gracefully)
        try:
            info = stock.info
            company_name = info.get("longName", info.get("shortName", symbol))
        except Exception:
            company_name = symbol

        current_price = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        exchange = "NSE" if ".NS" in ticker else "BSE"

        result = {
            "quote": {
                "symbol": symbol.upper(),
                "company_name": company_name,
                "exchange": exchange,
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "last_updated": data.index[-1].isoformat(),
            },
            "history": [
                {
                    "date": idx.isoformat(),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
                for idx, row in data.iterrows()
            ],
            "indicators": latest_indicators,
        }

        cache_service.set(cache_service.price_cache, cache_key, result)
        return result

    except StockNotFoundError:
        raise
    except Exception as e:
        raise ServiceUnavailableError("yfinance") from e


INTRADAY_INTERVAL_MAP = {
    "1m": {"interval": "1m", "period": "1d"},
    "5m": {"interval": "5m", "period": "5d"},
    "10m": {"interval": "5m", "period": "10d", "resample": "10T"},
    "20m": {"interval": "5m", "period": "20d", "resample": "20T"},
    "30m": {"interval": "30m", "period": "1mo"},
    "1h": {"interval": "60m", "period": "1mo"},
}


def fetch_intraday_data(symbol: str, interval: str = "5m") -> dict:
    """Fetch intraday OHLCV data for a symbol at the given interval.

    Supports: 1m, 5m, 10m, 20m, 30m, 1h.
    For intervals not natively supported (10m, 20m), data is fetched
    at 5m granularity and resampled server-side.
    """
    cache_key = f"intraday_{symbol}_{interval}"

    cached = cache_service.get(cache_service.price_cache, cache_key)
    if cached:
        return cached

    try:
        interval_config = INTRADAY_INTERVAL_MAP.get(interval)
        if not interval_config:
            interval_config = INTRADAY_INTERVAL_MAP["5m"]

        ticker = add_exchange_suffix(symbol)
        stock = yf.Ticker(ticker)
        data = stock.history(
            period=interval_config["period"],
            interval=interval_config["interval"],
        )

        if data.empty:
            raise StockNotFoundError(symbol)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].iloc[:, 0]
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data.dropna(inplace=True)

        # Resample if needed (for 10m, 20m)
        if "resample" in interval_config:
            resample_rule = interval_config["resample"]
            resampled = data.resample(resample_rule).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            resampled.dropna(inplace=True)
            data = resampled

        info = stock.info
        current_price = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100

        company_name = info.get("longName", info.get("shortName", symbol))
        exchange = "NSE" if ".NS" in ticker else "BSE"

        result = {
            "symbol": symbol.upper(),
            "company_name": company_name,
            "exchange": exchange,
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "last_updated": data.index[-1].isoformat(),
            "interval": interval,
            "history": [
                {
                    "time": idx.isoformat(),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
                for idx, row in data.iterrows()
            ],
        }

        cache_service.set(cache_service.price_cache, cache_key, result)
        return result

    except StockNotFoundError:
        raise
    except Exception as e:
        raise ServiceUnavailableError("yfinance") from e


def search_symbols(query: str, limit: int = 10) -> list[dict]:
    """Search for stock symbols by name or ticker."""
    try:
        ticker = yf.Ticker(query.upper())
        info = ticker.info
        if info and info.get("symbol"):
            name = info.get("longName", info.get("shortName", query.upper()))
            exchange = "NSE" if ".NS" in str(info.get("symbol", "")) else "BSE"
            return [
                {
                    "symbol": query.upper().replace(".NS", "").replace(".BO", ""),
                    "name": name,
                    "exchange": exchange,
                }
            ]
    except Exception:
        pass

    try:
        search_results = yf.Search(query)
        results = []
        for quote in search_results.quotes[:limit]:
            symbol = quote.get("symbol", "")
            exchange = "NSE" if ".NS" in symbol else "BSE"
            results.append(
                {
                    "symbol": symbol.replace(".NS", "").replace(".BO", ""),
                    "name": quote.get("shortname", quote.get("longname", symbol)),
                    "exchange": exchange,
                }
            )
        return results
    except Exception:
        return []
