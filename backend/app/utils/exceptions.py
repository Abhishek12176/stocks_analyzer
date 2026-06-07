from fastapi import HTTPException


class StockNotFoundError(HTTPException):
    def __init__(self, symbol: str):
        super().__init__(
            status_code=404,
            detail={
                "error": f"Stock '{symbol}' not found",
                "code": "STOCK_NOT_FOUND",
                "details": f"No data available for symbol '{symbol}'. Verify the symbol is correct and try again.",
            },
        )


class ServiceUnavailableError(HTTPException):
    def __init__(self, service: str):
        super().__init__(
            status_code=503,
            detail={
                "error": f"{service} is temporarily unavailable",
                "code": "SERVICE_UNAVAILABLE",
                "details": f"The data source '{service}' could not be reached. Please try again later.",
            },
        )


class RateLimitError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED",
                "details": "Too many requests. Please wait before trying again.",
            },
        )


class InvalidSymbolError(HTTPException):
    def __init__(self, symbol: str):
        super().__init__(
            status_code=400,
            detail={
                "error": f"Invalid symbol: '{symbol}'",
                "code": "INVALID_SYMBOL",
                "details": "Symbol must be 1-10 alphanumeric characters.",
            },
        )
