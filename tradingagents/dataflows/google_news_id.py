"""Google News RSS source localized for Indonesia (hl=id, gl=ID).

No API key required. Uses the public Google News RSS endpoint.
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from dateutil.relativedelta import relativedelta

from .idx_news_rss import fetch_idx_rss, filter_by_keywords
from .market_utils import global_news_queries, strip_market_suffix


_RSS_URL = "https://news.google.com/rss/search"
_YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
_HL = "id"
_GL = "ID"
_CEID = "ID:id"
_TIMEOUT = 15


def _fetch_rss(query: str, limit: int) -> list[dict]:
    params = {
        "q": query,
        "hl": _HL,
        "gl": _GL,
        "ceid": _CEID,
    }
    url = f"{_RSS_URL}?{urllib.parse.urlencode(params)}"
    resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    items = []
    root = ET.fromstring(resp.content)
    for item in list(root.iterfind(".//item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        source_el = item.find("source")
        publisher = (source_el.text.strip() if source_el is not None and source_el.text else "Google News")
        description = (item.findtext("description") or "").strip()

        pub_date = None
        if pub:
            try:
                pub_date = parsedate_to_datetime(pub)
            except (TypeError, ValueError):
                pub_date = None

        items.append({
            "title": title,
            "link": link,
            "publisher": publisher,
            "summary": description,
            "pub_date": pub_date,
        })
    return items


def _fetch_yahoo_rss(symbol: str, limit: int) -> list[dict]:
    """Fetch Yahoo Finance per-ticker RSS headlines."""
    url = f"{_YAHOO_RSS_URL}?{urllib.parse.urlencode({'s': symbol, 'region': 'US', 'lang': 'en-US'})}"
    resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    items = []
    root = ET.fromstring(resp.content)
    for item in list(root.iterfind(".//item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        description = (item.findtext("description") or "").strip()

        pub_date = None
        if pub:
            try:
                pub_date = parsedate_to_datetime(pub)
            except (TypeError, ValueError):
                pub_date = None

        items.append({
            "title": title,
            "link": link,
            "publisher": "Yahoo Finance",
            "summary": description,
            "pub_date": pub_date,
        })
    return items


def _format_articles(articles: list[dict]) -> str:
    out = ""
    for a in articles:
        out += f"### {a['title']} (source: {a['publisher']})\n"
        if a.get("summary"):
            out += f"{a['summary']}\n"
        if a.get("pub_date"):
            out += f"Published: {a['pub_date'].strftime('%Y-%m-%d')}\n"
        if a.get("link"):
            out += f"Link: {a['link']}\n"
        out += "\n"
    return out


def _within_range(article: dict, start_dt: datetime, end_dt: datetime) -> bool:
    pub = article.get("pub_date")
    if not pub:
        return True  # keep undated items rather than drop silently
    pub_naive = pub.astimezone(timezone.utc).replace(tzinfo=None) if pub.tzinfo else pub
    return start_dt <= pub_naive <= end_dt + relativedelta(days=1)


def get_news_google_id(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch ticker-specific news from Google News (Indonesian locale).

    Many IDX tickers are short codes that collide with English words (e.g. AUTO,
    BIRD, GOTO). To improve recall, query several variants and merge by URL.
    """
    bare = strip_market_suffix(ticker)
    suffixed = bare if "." in ticker else f"{bare}.JK"
    queries = [
        f'"{bare}.JK"',
        f'"saham {bare}"',
        f'"{bare}" IDX OR IHSG OR BEI',
        f'"{bare}" saham emiten',
        f'"{bare}" site:idx.co.id',
        f'"{bare}" "keterbukaan informasi"',
    ]

    seen_keys: set[str] = set()
    merged: list[dict] = []
    errors = 0
    total_sources = len(queries) + 1  # +1 for Yahoo

    for q in queries:
        try:
            for a in _fetch_rss(q, limit=25):
                key = a.get("link") or a.get("title")
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(a)
        except Exception:
            errors += 1
            continue

    # Yahoo Finance per-ticker headlines (English; complements Indonesian sources).
    try:
        for a in _fetch_yahoo_rss(suffixed, limit=20):
            key = a.get("link") or a.get("title")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(a)
    except Exception:
        errors += 1

    # Native Indonesian publishers (Bisnis, Kontan, CNBC Indonesia, Detik,
    # Investor.id, Emiten News). Pull the latest items from each feed and
    # keep only those that mention the ticker — these often surface
    # corporate actions and regulatory news days before Google News.
    try:
        rss_articles, rss_errs = fetch_idx_rss(per_feed_limit=25)
        ticker_kw = [bare, f"saham {bare}", f"{bare}.JK", f"{bare}.jk"]
        for a in filter_by_keywords(rss_articles, ticker_kw):
            key = a.get("link") or a.get("title")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(a)
        # Count failed-feed loads as a single source error so the
        # "all sources errored" guard below can still trigger.
        if rss_errs and not rss_articles:
            errors += 1
    except Exception:
        errors += 1

    if not merged:
        if errors == total_sources:
            return f"Error fetching news for {ticker}"
        return f"No news found for {ticker}"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    filtered = [a for a in merged if _within_range(a, start_dt, end_dt)]

    if not filtered:
        return f"No news found for {ticker} between {start_date} and {end_date}"

    # Sort newest first when dates are available.
    filtered.sort(key=lambda a: a.get("pub_date") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return f"## {ticker} News (Google News ID), from {start_date} to {end_date}:\n\n{_format_articles(filtered)}"


def get_global_news_google_id(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    """Fetch macro/global news using market-localized queries."""
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)

    seen = set()
    collected: list[dict] = []
    try:
        for query in global_news_queries():
            for a in _fetch_rss(query, limit=limit):
                key = a["title"]
                if not key or key in seen:
                    continue
                if not _within_range(a, start_dt, curr_dt):
                    continue
                seen.add(key)
                collected.append(a)
                if len(collected) >= limit:
                    break
            if len(collected) >= limit:
                break
    except Exception as e:
        return f"Error fetching global news (ID): {e}"

    # Top up with native Indonesian market headlines (unfiltered, latest
    # from each feed). These give better macro/sector context for IHSG
    # than Google News alone.
    if len(collected) < limit:
        try:
            rss_articles, _errs = fetch_idx_rss(per_feed_limit=10)
            for a in rss_articles:
                key = a.get("title") or a.get("link")
                if not key or key in seen:
                    continue
                if not _within_range(a, start_dt, curr_dt):
                    continue
                seen.add(key)
                collected.append(a)
                if len(collected) >= limit:
                    break
        except Exception:
            pass

    if not collected:
        return f"No global news found for {curr_date}"

    start_str = start_dt.strftime("%Y-%m-%d")
    return f"## Global Market News (Google News ID), from {start_str} to {curr_date}:\n\n{_format_articles(collected[:limit])}"
