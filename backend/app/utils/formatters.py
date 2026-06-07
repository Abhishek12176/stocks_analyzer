from datetime import datetime, timezone


def format_date(dt: datetime | None) -> str:
    if not dt:
        return "Latest"
    return dt.strftime("%d %b %Y, %I:%M %p")


def parse_iso_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
