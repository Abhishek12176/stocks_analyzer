import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, SMA20, SMA50 to a DataFrame with OHLCV columns."""
    df = df.copy()
    df["RSI"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["Signal"] = macd["MACDs_12_26_9"]
    df["SMA20"] = ta.sma(df["Close"], length=20)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """Extract the latest indicator values from a DataFrame."""
    latest = df.iloc[-1]
    return {
        "rsi": float(latest["RSI"]) if pd.notna(latest["RSI"]) else None,
        "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
        "signal": float(latest["Signal"]) if pd.notna(latest["Signal"]) else None,
        "sma20": float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else None,
        "sma50": float(latest["SMA50"]) if pd.notna(latest["SMA50"]) else None,
    }
