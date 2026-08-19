import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Histogram": histogram,
    })


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = rsi(df["Close"], length=14)
    macd_df = macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd_df["MACD"]
    df["Signal"] = macd_df["Signal"]
    df["SMA20"] = sma(df["Close"], length=20)
    df["SMA50"] = sma(df["Close"], length=50)
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    return {
        "rsi": float(latest["RSI"]) if pd.notna(latest["RSI"]) else None,
        "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
        "signal": float(latest["Signal"]) if pd.notna(latest["Signal"]) else None,
        "sma20": float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else None,
        "sma50": float(latest["SMA50"]) if pd.notna(latest["SMA50"]) else None,
    }
