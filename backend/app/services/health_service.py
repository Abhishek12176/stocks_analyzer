import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import requests

from app.config import settings

logger = logging.getLogger("equitylens.health")

SOURCE_TIMEOUT = 15


class SourceHealth:
    name: str
    status: str
    latency_ms: float
    error: Optional[str]
    last_checked: str

    def __init__(self, name: str, status: str, latency_ms: float, error: Optional[str] = None):
        self.name = name
        self.status = status
        self.latency_ms = round(latency_ms, 1)
        self.error = error
        self.last_checked = datetime.now(timezone.utc).isoformat()

    def dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "last_checked": self.last_checked,
        }


class SourceHealthChecker:

    async def check_yfinance(self) -> SourceHealth:
        start = time.time()
        try:
            import yfinance as yf
            ticker = yf.Ticker("RELIANCE.NS")
            info = ticker.info or {}
            if info.get("symbol") or info.get("regularMarketPrice"):
                return SourceHealth("yfinance", "ok", (time.time() - start) * 1000)
            return SourceHealth("yfinance", "error", (time.time() - start) * 1000, "No data returned")
        except Exception as e:
            return SourceHealth("yfinance", "error", (time.time() - start) * 1000, str(e))

    async def check_screener(self) -> SourceHealth:
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.screener.in/",
                    timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                ) as resp:
                    if resp.status == 200:
                        return SourceHealth("screener.in", "ok", (time.time() - start) * 1000)
                    return SourceHealth("screener.in", "error", (time.time() - start) * 1000, f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return SourceHealth("screener.in", "error", (time.time() - start) * 1000, "Timeout")
        except Exception as e:
            return SourceHealth("screener.in", "error", (time.time() - start) * 1000, str(e))

    async def check_marketsmith(self) -> SourceHealth:
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://marketsmithindia.com/",
                    timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                ) as resp:
                    if resp.status == 200:
                        return SourceHealth("marketsmithindia.com", "ok", (time.time() - start) * 1000)
                    return SourceHealth("marketsmithindia.com", "error", (time.time() - start) * 1000, f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return SourceHealth("marketsmithindia.com", "error", (time.time() - start) * 1000, "Timeout")
        except Exception as e:
            return SourceHealth("marketsmithindia.com", "error", (time.time() - start) * 1000, str(e))

    async def check_moneycontrol(self) -> SourceHealth:
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.moneycontrol.com/",
                    timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                ) as resp:
                    if resp.status == 200:
                        return SourceHealth("moneycontrol.com", "ok", (time.time() - start) * 1000)
                    return SourceHealth("moneycontrol.com", "error", (time.time() - start) * 1000, f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return SourceHealth("moneycontrol.com", "error", (time.time() - start) * 1000, "Timeout")
        except Exception as e:
            return SourceHealth("moneycontrol.com", "error", (time.time() - start) * 1000, str(e))

    async def check_nse(self) -> SourceHealth:
        start = time.time()
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.nseindia.com/",
            })
            session.get("https://www.nseindia.com/", timeout=SOURCE_TIMEOUT)
            resp = session.get(
                "https://www.nseindia.com/api/equity-master",
                timeout=SOURCE_TIMEOUT,
                params={"state": "active"},
            )
            if resp.status_code == 200:
                return SourceHealth("nseindia.com", "ok", (time.time() - start) * 1000)
            return SourceHealth("nseindia.com", "error", (time.time() - start) * 1000, f"HTTP {resp.status_code}")
        except Exception as e:
            return SourceHealth("nseindia.com", "error", (time.time() - start) * 1000, str(e))

    async def check_newsdata(self) -> SourceHealth:
        start = time.time()
        if not settings.newsdata_api_key:
            return SourceHealth("newsdata.io", "ok", 0, "No API key configured")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://newsdata.io/api/1/news",
                    params={"apikey": settings.newsdata_api_key, "q": "test", "size": 1},
                    timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "success":
                            return SourceHealth("newsdata.io", "ok", (time.time() - start) * 1000)
                        return SourceHealth("newsdata.io", "error", (time.time() - start) * 1000, data.get("status", "Unknown"))
                    return SourceHealth("newsdata.io", "error", (time.time() - start) * 1000, f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return SourceHealth("newsdata.io", "error", (time.time() - start) * 1000, "Timeout")
        except Exception as e:
            return SourceHealth("newsdata.io", "error", (time.time() - start) * 1000, str(e))

    async def check_all(self) -> list[SourceHealth]:
        checks = [
            self.check_yfinance(),
            self.check_screener(),
            self.check_marketsmith(),
            self.check_moneycontrol(),
            self.check_nse(),
            self.check_newsdata(),
        ]
        results = await asyncio.gather(*checks, return_exceptions=True)
        output = []
        for r in results:
            if isinstance(r, SourceHealth):
                output.append(r)
            else:
                output.append(SourceHealth("unknown", "error", 0, str(r)))
        unhealthy = [s.name for s in output if s.status == "error"]
        if unhealthy:
            logger.warning("Unhealthy data sources detected: %s", ", ".join(unhealthy))
        return output


health_checker = SourceHealthChecker()
