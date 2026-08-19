import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_watchlist() -> list[str]:
    _ensure_data_dir()
    if not os.path.exists(WATCHLIST_FILE):
        return []
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_watchlist(symbols: list[str]):
    _ensure_data_dir()
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(symbols, f, indent=2)


@router.get("/")
async def get_watchlist():
    """Get the list of watched symbols."""
    return {"symbols": _load_watchlist()}


class WatchlistAddRequest(BaseModel):
    symbol: str


@router.post("/{symbol}")
async def add_to_watchlist(symbol: str):
    """Add a symbol to the watchlist."""
    clean = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
    symbols = _load_watchlist()
    if clean not in symbols:
        symbols.append(clean)
        _save_watchlist(symbols)
    return {"symbols": symbols}


@router.delete("/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Remove a symbol from the watchlist."""
    clean = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
    symbols = _load_watchlist()
    if clean in symbols:
        symbols.remove(clean)
        _save_watchlist(symbols)
    return {"symbols": symbols}
