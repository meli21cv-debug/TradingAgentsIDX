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


def _undouble(ticker: str) -> str:
    """Detect accidental ticker doubling like 'RAJARAJA' -> 'RAJA'.

    Some agent flows have echoed the symbol twice when constructing tool
    arguments. If the string is exactly two identical halves of length 3-6,
    collapse it. This is a narrow guard — bare doubles outside that length
    range are left alone to avoid false positives.
    """
    if not ticker or "." in ticker:
        return ticker
    n = len(ticker)
    if n % 2 == 0 and 6 <= n <= 12:
        half = n // 2
        if ticker[:half] == ticker[half:]:
            return ticker[:half]
    return ticker


def normalize_ticker(ticker: str) -> str:
    """Trim, uppercase, and de-duplicate the ticker symbol."""
    if not ticker:
        return ticker
    cleaned = ticker.strip().upper()
    # Split into bare + suffix, undouble bare half only.
    if "." in cleaned:
        base, suffix = cleaned.split(".", 1)
        return f"{_undouble(base)}.{suffix}"
    return _undouble(cleaned)


def apply_market_suffix(ticker: str) -> str:
    """Normalize the ticker and append the configured market suffix if absent."""
    ticker = normalize_ticker(ticker)
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
