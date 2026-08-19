import re

NSE_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,10}$")


def clean_symbol(symbol: str) -> str:
    """Normalize a stock symbol by removing exchange suffixes and uppercasing."""
    return (
        symbol.upper()
        .replace(".NS", "")
        .replace(".NSE", "")
        .replace(".BO", "")
        .replace(".BSE", "")
        .strip()
    )


def validate_symbol(symbol: str) -> bool:
    """Check if a symbol matches the NSE/BSE stock symbol pattern."""
    return bool(NSE_SYMBOL_PATTERN.match(symbol))


def add_exchange_suffix(symbol: str, exchange: str = "NS") -> str:
    """Append the yfinance exchange suffix to a symbol."""
    clean = clean_symbol(symbol)
    if exchange.upper() == "BSE":
        return f"{clean}.BO"
    return f"{clean}.NS"
