"""Point-in-time (PIT) event layer for the 20-day forecasting system (Task 6).

Events are discrete corporate/market/earnings/news facts. To guarantee the
no-lookahead invariant every event carries three timestamps:

- `event_timestamp`     : when the event actually happened.
- `available_timestamp` : (optional) when it became public/knowable. When
                          present it overrides `event_timestamp` for the
                          visibility decision.
- `effective_date`      : (optional) the explicit day from which the event may
                          influence a prediction. When omitted it is derived
                          from the event time:
                              same calendar day if at/before the NSE close
                              (default local 15:30 IST, but configurable),
                              otherwise the *next* day.

An event is visible for a day D iff  D >= effective_date. In particular an
event that happens *after the close of day T* gets `effective_date = T+1` and
can therefore never influence the day-T prediction row. This is exactly the
invariant asserted by the Task-6 PIT test.

Event-feature columns (causal, built from effective dates only):
- `event_count_{window}d` : # events effective within the trailing window
- `event_today`           : 1 if any event is effective on this day, else 0

Events whose effective date falls on a non-trading calendar day (weekend,
holiday) are preserved inside the running cumulative count, so they surface
at the next trading date (never dropped, never backdated).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pandas as pd

from app.config import settings

IST_OFFSET = timedelta(hours=5, minutes=30)
EVENT_CUTOFF_LOCAL = time(15, 30)
EVENT_WINDOW = 5

_EVENT_TIMESTAMPS = ("event_timestamp", "available_timestamp")
_EVENT_STRINGS = ("category", "title", "subtype")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _to_datetime(value) -> datetime:
    if value is None:
        raise ValueError("event is missing a timestamp value")
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _to_local_ist(value) -> datetime:
    """Return a timezone-naive IST wall-clock datetime for `value`.

    Timezone-aware inputs are converted to UTC then offset by +05:30, so an
    instant that is 19:00 IST is treated as an evening event regardless of
    whether it was supplied as UTC or IST. Naive inputs are assumed to already
    be IST wall-clock (the NSE-native convention).
    """
    dt = _to_datetime(value)
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt
    return (dt.astimezone(timezone.utc) + IST_OFFSET).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Event normalisation + effective date
# ---------------------------------------------------------------------------

def effective_date(
    event: dict,
    cutoff_time: time = EVENT_CUTOFF_LOCAL,
) -> pd.Timestamp:
    """First day an event may influence a prediction (no-lookahead).

    Priority: explicit `effective_date` >> `available_timestamp` >>
    `event_timestamp` (visibility time). If the computed event time is
    `<= cutoff_time` (default 15:30 IST = NSE close) it is effective the same
    day; otherwise the next calendar day.
    """
    if event.get("effective_date") is not None:
        return pd.Timestamp(event["effective_date"]).normalize()

    known_at = None
    for key in ("available_timestamp", "event_timestamp"):
        if event.get(key) is not None:
            known_at = event[key]
            break
    if known_at is None:
        raise ValueError(
            "event must carry event_timestamp, available_timestamp or effective_date"
        )

    local = _to_local_ist(known_at)
    day = local.date()
    if local.time() <= cutoff_time:
        return pd.Timestamp(day)
    return pd.Timestamp(day + timedelta(days=1))


def is_visible(event: dict, day: pd.Timestamp | datetime | str) -> bool:
    """True if the event is knowable on/at `day` (PIT-safe)."""
    return pd.Timestamp(day).normalize() >= effective_date(event)


def filter_visible(events: list[dict], as_of: pd.Timestamp | datetime | str) -> list[dict]:
    """Return the subset of events knowable as of `as_of` (no lookahead)."""
    as_of_norm = pd.Timestamp(as_of).normalize()
    return [e for e in events if pd.Timestamp(effective_date(e)).normalize() <= as_of_norm]


# ---------------------------------------------------------------------------
# Event features
# ---------------------------------------------------------------------------

def list_event_features(window: int = EVENT_WINDOW) -> list[str]:
    return sorted({f"event_count_{window}d", "event_today"})


def _daily_counts(
    events: list[dict],
    index: pd.DatetimeIndex,
    pad_days: int = 0,
) -> pd.Series:
    """Per-calendar-day event counts over the full daily range (not the sparse
    trading index) so weekend/holiday events survive into the running sum.

    The calendar is padded `pad_days` to the front so trailing-window counts
    on the earliest rows already see events that predate the frame.
    """
    if not index.empty:
        lo = index.min().normalize()
        hi = index.max().normalize()
        full = pd.date_range(lo - pd.Timedelta(days=pad_days), hi, freq="D")
    else:
        full = pd.DatetimeIndex([])

    eff = [effective_date(e) for e in events]
    counts = pd.Series(eff).value_counts().sort_index()
    daily = pd.Series(0, index=full, dtype=int)
    daily = daily.add(counts.reindex(full).fillna(0))
    return daily.astype(int)


def add_event_features(
    df: pd.DataFrame,
    events: list[dict],
    window: int = EVENT_WINDOW,
) -> pd.DataFrame:
    """Attach count-of-events features to a copy of the frame (causal)."""
    out = df.copy()
    trading = pd.DatetimeIndex(out.index).normalize()

    daily = _daily_counts(events, trading, pad_days=window)
    cum = daily.cumsum()
    cum_traded = cum.reindex(trading).fillna(0).astype(int)

    out[f"event_count_{window}d"] = (cum_traded - cum_traded.shift(window, fill_value=0)).astype(int)
    traded_today = (cum_traded - cum_traded.shift(1, fill_value=0)).astype(int)
    out["event_today"] = (traded_today > 0).astype(int)

    out.attrs["feature_version"] = settings.ml_feature_version
    return out