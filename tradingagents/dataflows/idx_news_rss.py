"""Indonesian financial-news RSS aggregator.

Pulls headlines from native Indonesian publishers (Bisnis, Kontan, CNBC
Indonesia, Detik Finance, Investor.id, Emiten News). Items are returned
in the same dict shape that :mod:`google_news_id` produces, so the
two sources can be merged transparently by URL.

No API keys required. Each feed is tried independently; transient
failures bump the error counter but never abort the call.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

_TIMEOUT = 12
_UA = {"User-Agent": "Mozilla/5.0 (compatible; TradingAgents/0.2)"}

# (publisher_label, rss_url) — ordered by coverage / freshness.
_IDX_FEEDS: tuple[tuple[str, str], ...] = (
    ("Bisnis Market",    "https://market.bisnis.com/rss"),
    ("Bisnis Finansial", "https://finansial.bisnis.com/rss"),
    ("Kontan Investasi", "https://investasi.kontan.co.id/rss"),
    ("Kontan Keuangan",  "https://keuangan.kontan.co.id/rss"),
    ("Kontan Industri",  "https://industri.kontan.co.id/rss"),
    ("Detik Finance",    "https://rss.detik.com/index.php/finance"),
    ("CNBC Indonesia",   "https://www.cnbcindonesia.com/market/rss"),
    ("Investor.id",      "https://investor.id/rss"),
    ("Emiten News",      "https://emitennews.com/rss"),
)


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _parse_rss(content: bytes, publisher: str, limit: int) -> list[dict]:
    items: list[dict] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for item in list(root.iterfind(".//item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        description = _strip_html(item.findtext("description") or "")

        pub_date = None
        if pub:
            try:
                pub_date = parsedate_to_datetime(pub)
            except (TypeError, ValueError):
                pub_date = None

        if not title or not link:
            continue
        items.append({
            "title": title,
            "link": link,
            "publisher": publisher,
            "summary": description,
            "pub_date": pub_date,
        })
    return items


def fetch_idx_rss(per_feed_limit: int = 20) -> tuple[list[dict], int]:
    """Fetch latest items across all configured IDX feeds.

    Returns ``(articles, error_count)``. Articles are NOT filtered by
    ticker — callers should filter by keyword match.
    """
    articles: list[dict] = []
    errors = 0
    for publisher, url in _IDX_FEEDS:
        try:
            resp = requests.get(url, timeout=_TIMEOUT, headers=_UA)
            resp.raise_for_status()
            articles.extend(_parse_rss(resp.content, publisher, per_feed_limit))
        except Exception:
            errors += 1
            continue
    return articles, errors


def filter_by_keywords(articles: list[dict], keywords: list[str]) -> list[dict]:
    """Keep articles whose title or summary contains any keyword (case-insensitive)."""
    needles = [k.lower() for k in keywords if k]
    if not needles:
        return articles
    kept = []
    for a in articles:
        hay = ((a.get("title") or "") + " " + (a.get("summary") or "")).lower()
        if any(n in hay for n in needles):
            kept.append(a)
    return kept
