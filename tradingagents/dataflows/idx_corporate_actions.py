"""Indonesian corporate-action calendar.

Two sources stacked:

  1. yahooquery — dividend history, splits, upcoming earnings / ex-div
     dates. Reliable for the .JK names Yahoo covers.
  2. Native Indonesian RSS (via :mod:`idx_news_rss`) — surfaces RUPS,
     cum-/ex-date, rights-issue, buyback, and M&A announcements that
     may precede Yahoo's calendar.

Returns formatted markdown over a ±``window_days`` window centred on
``curr_date``. The two layers are independent: if one fails the other
still renders.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from yahooquery import Ticker

from .idx_news_rss import fetch_idx_rss, filter_by_keywords
from .market_utils import strip_market_suffix

# Theme keywords that mark a story as a corporate action (Indonesian +
# English). Lower-cased; matched against title + summary.
_CA_THEMES = (
    "rups", "dividen", "cum date", "ex date", "ex-date", "cum-date",
    "stock split", "right issue", "rights issue", "buyback", "buy back",
    "akuisisi", "divestasi", "merger", "tender offer", "tender penawaran",
    "ipo", "go public", "listing", "delisting", "spin off", "spin-off",
    "private placement", "obligasi", "warrant", "kuasi reorganisasi",
)


def _format_history_df(df, label: str, max_rows: int = 8) -> str:
    if df is None or not hasattr(df, "empty") or df.empty:
        return ""
    return f"### {label}\n```\n{df.head(max_rows).to_string()}\n```\n"


def _yahoo_block(ticker_u: str) -> list[str]:
    parts: list[str] = []
    try:
        t = Ticker(ticker_u)
    except Exception as e:  # noqa: BLE001
        return [f"_yahooquery init failed: {e}_"]

    # Calendar events (upcoming earnings, ex-dividend)
    try:
        cal = t.calendar_events
        if isinstance(cal, dict):
            cal = cal.get(ticker_u, cal)
        if isinstance(cal, dict) and cal:
            lines = []
            ex_div = cal.get("exDividendDate")
            if ex_div:
                lines.append(f"- Ex-dividend date: {ex_div}")
            earnings = cal.get("earnings") or {}
            if isinstance(earnings, dict):
                e_date = (
                    earnings.get("earningsDate")
                    or earnings.get("earningsAverage")
                )
                if e_date:
                    lines.append(f"- Next earnings: {e_date}")
            if lines:
                parts.append("## Upcoming Calendar (yahooquery)")
                parts.extend(lines)
                parts.append("")
    except Exception:  # noqa: BLE001
        pass

    # Dividend history
    try:
        dividend_df = t.dividend_history(start="2020-01-01")
        block = _format_history_df(dividend_df, "Dividend History (yahooquery)")
        if block:
            parts.append(block)
    except Exception:  # noqa: BLE001
        pass

    # Split history via the events flag
    try:
        splits = t.history(period="max", interval="1d", events="splits")
        if hasattr(splits, "empty") and not splits.empty and "splits" in splits.columns:
            splits = splits[splits["splits"] != 0][["splits"]]
            block = _format_history_df(splits, "Stock Splits (yahooquery)")
            if block:
                parts.append(block)
    except Exception:  # noqa: BLE001
        pass

    return parts


def _rss_block(bare: str, curr_date: str | None, window_days: int) -> list[str]:
    parts: list[str] = []
    try:
        articles, _errs = fetch_idx_rss(per_feed_limit=30)
    except Exception as e:  # noqa: BLE001
        return [f"_Indonesian RSS scan failed: {e}_"]

    ticker_kw = [bare, f"saham {bare}", f"{bare}.jk"]
    candidates = filter_by_keywords(articles, ticker_kw)
    scoped = filter_by_keywords(candidates, list(_CA_THEMES))

    if curr_date:
        try:
            centre = datetime.strptime(curr_date, "%Y-%m-%d")
            cutoff = centre - timedelta(days=window_days)
            def _in_window(a):
                pub = a.get("pub_date")
                if not pub:
                    return True
                pub_naive = pub.replace(tzinfo=None) if pub.tzinfo else pub
                return pub_naive >= cutoff
            scoped = [a for a in scoped if _in_window(a)]
        except ValueError:
            pass

    scoped.sort(
        key=lambda a: a.get("pub_date") or datetime.min,
        reverse=True,
    )

    if not scoped:
        parts.append(
            "## Corporate-Action Announcements\n"
            "_No matching headlines in Indonesian RSS scan._\n"
        )
        return parts

    parts.append(f"## Corporate-Action Announcements ({len(scoped)} hits, Indonesian RSS)")
    for a in scoped[:15]:
        date = a["pub_date"].strftime("%Y-%m-%d") if a.get("pub_date") else "n/a"
        parts.append(f"- {date} [{a['publisher']}] {a['title']}")
        if a.get("link"):
            parts.append(f"  {a['link']}")
    parts.append("")
    return parts


def get_corporate_actions(
    ticker: Annotated[str, "ticker symbol (may include .JK)"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
    window_days: Annotated[int, "± window in days for RSS scan"] = 90,
) -> str:
    """Return corporate-action context: dividends, splits, earnings dates,
    plus near-term Indonesian-news mentions of RUPS / rights / buyback / M&A.
    """
    ticker_u = ticker.upper()
    bare = strip_market_suffix(ticker_u)

    header = [f"# Corporate Actions — {ticker_u}"]
    if curr_date:
        header.append(f"_Window: ±{window_days} days around {curr_date}_")
    header.append("")

    body = _yahoo_block(ticker_u) + _rss_block(bare, curr_date, window_days)
    return "\n".join(header + body).rstrip() + "\n"
