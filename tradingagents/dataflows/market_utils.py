"""Market-specific helpers for ticker normalization and news queries."""

from .config import get_config


_MARKET_SUFFIX = {
    "ID": ".JK",
}

_GLOBAL_QUERIES = {
    "ID": [
        "IHSG bursa efek Indonesia",
        "Bank Indonesia suku bunga",
        "rupiah ekonomi Indonesia",
        "pasar saham IDX",
    ],
    None: [
        "stock market economy",
        "Federal Reserve interest rates",
        "inflation economic outlook",
        "global markets trading",
    ],
}


def get_market() -> str | None:
    return get_config().get("market")


def apply_market_suffix(ticker: str) -> str:
    """Append the configured market suffix when the ticker has none."""
    if not ticker or "." in ticker:
        return ticker
    suffix = _MARKET_SUFFIX.get(get_market() or "")
    return f"{ticker}{suffix}" if suffix else ticker


def strip_market_suffix(ticker: str) -> str:
    """Return the bare symbol without an exchange suffix (e.g. BBCA.JK -> BBCA)."""
    return ticker.split(".", 1)[0] if ticker else ticker


def global_news_queries() -> list[str]:
    market = get_market()
    return _GLOBAL_QUERIES.get(market, _GLOBAL_QUERIES[None])
