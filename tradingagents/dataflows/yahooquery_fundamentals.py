"""yahooquery-based fundamentals fetchers.

yahooquery hits Yahoo Finance's `fundamentals/timeseries` endpoint, which
typically returns more periods and more line items than yfinance's
`Ticker.financials` family. Used as the primary fundamentals vendor with
yfinance as the fallback (configured via data_vendors.fundamental_data).

Output format matches the yfinance vendor: CSV string with a header line.
The yahooquery DataFrame is pivoted from its native long format
(rows=periods, cols=line items) to wide format (rows=line items,
cols=periods) so downstream agents see the same shape as the yfinance
output.
"""

from datetime import datetime
from typing import Annotated, Callable

import pandas as pd
from yahooquery import Ticker


def _retry(fn: Callable, attempts: int = 2):
    """Tiny retry helper — yahooquery occasionally returns transient errors."""
    last_err = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err  # type: ignore[misc]


def _pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yahooquery long-format (asOfDate as a column, line items as
    columns) to wide format (line items as rows, periods as columns).

    Drops the helper columns periodType and currencyCode after using them
    to label the period column.
    """
    if df is None or not hasattr(df, "empty") or df.empty:
        return pd.DataFrame()
    if "asOfDate" not in df.columns:
        return df  # unexpected shape, hand it back as-is
    df = df.copy()
    # Build column labels like "2026-03-31" or "2026-03-31 (TTM)"
    period_label = df["asOfDate"].astype(str)
    if "periodType" in df.columns:
        period_label = period_label + df["periodType"].apply(
            lambda p: f" ({p})" if isinstance(p, str) and p and p != "3M" and p != "12M" else ""
        )
    drop_cols = [c for c in ("asOfDate", "periodType", "currencyCode") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df.index = period_label
    # Rows are now periods; transpose so rows = line items, cols = periods.
    wide = df.T
    # Sort columns chronologically (most recent first to match yfinance default).
    try:
        wide = wide.reindex(sorted(wide.columns, reverse=True), axis=1)
    except Exception:
        pass
    return wide


def _filter_by_curr_date(df: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    """Drop columns whose date is later than curr_date (look-ahead guard).

    Mirrors filter_financials_by_date from y_finance.py.
    """
    if df is None or df.empty or not curr_date:
        return df
    try:
        cutoff = pd.Timestamp(curr_date)
    except Exception:
        return df
    keep = []
    for col in df.columns:
        try:
            # period_label may be "YYYY-MM-DD" or "YYYY-MM-DD (annual)" — split.
            d = str(col).split(" ", 1)[0]
            ts = pd.Timestamp(d)
            if ts <= cutoff:
                keep.append(col)
        except Exception:
            keep.append(col)  # keep if unparseable rather than drop silently
    return df[keep]


# ---------------------------------------------------------------------------
# Public fetchers — same signatures as the yfinance vendor.
# ---------------------------------------------------------------------------


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date (YYYY-MM-DD); not used here"] = None,
) -> str:
    """High-level fundamentals snapshot via yahooquery.

    Combines summary_detail + key_stats + financial_data into the same
    label/value shape the yfinance vendor produces, but with more fields
    when Yahoo provides them.
    """
    try:
        ticker_u = ticker.upper()
        t = Ticker(ticker_u)
        sd = _retry(lambda: t.summary_detail).get(ticker_u, {})
        ks = _retry(lambda: t.key_stats).get(ticker_u, {})
        fd = _retry(lambda: t.financial_data).get(ticker_u, {})
        ap = _retry(lambda: t.asset_profile).get(ticker_u, {})
        price = _retry(lambda: t.price).get(ticker_u, {})

        # Some endpoints return a string ("Quote not found") on failure.
        if not isinstance(sd, dict) and not isinstance(ks, dict):
            return f"No fundamentals data found for symbol '{ticker}'"

        sd = sd if isinstance(sd, dict) else {}
        ks = ks if isinstance(ks, dict) else {}
        fd = fd if isinstance(fd, dict) else {}
        ap = ap if isinstance(ap, dict) else {}
        price = price if isinstance(price, dict) else {}

        fields = [
            ("Name", price.get("longName") or price.get("shortName")),
            ("Sector", ap.get("sector")),
            ("Industry", ap.get("industry")),
            ("Market Cap", price.get("marketCap")),
            ("Currency", price.get("currency")),
            ("Regular Market Price", price.get("regularMarketPrice")),
            ("PE Ratio (TTM)", sd.get("trailingPE")),
            ("Forward PE", sd.get("forwardPE") or ks.get("forwardPE")),
            ("PEG Ratio", ks.get("pegRatio")),
            ("Price to Book", ks.get("priceToBook") or sd.get("priceToBook")),
            ("Price to Sales (TTM)", sd.get("priceToSalesTrailing12Months")),
            ("EV/EBITDA", ks.get("enterpriseToEbitda")),
            ("EV/Revenue", ks.get("enterpriseToRevenue")),
            ("EPS (TTM)", ks.get("trailingEps")),
            ("Forward EPS", ks.get("forwardEps")),
            ("Dividend Rate", sd.get("dividendRate")),
            ("Dividend Yield", sd.get("dividendYield")),
            ("Payout Ratio", sd.get("payoutRatio")),
            ("5y Avg Dividend Yield", sd.get("fiveYearAvgDividendYield")),
            ("Beta", sd.get("beta") or ks.get("beta")),
            ("52 Week High", sd.get("fiftyTwoWeekHigh") or ks.get("fiftyTwoWeekHigh")),
            ("52 Week Low", sd.get("fiftyTwoWeekLow") or ks.get("fiftyTwoWeekLow")),
            ("50 Day Average", ks.get("fiftyDayAverage")),
            ("200 Day Average", ks.get("twoHundredDayAverage")),
            ("Revenue (TTM)", fd.get("totalRevenue")),
            ("Revenue Per Share", fd.get("revenuePerShare")),
            ("Revenue Growth", fd.get("revenueGrowth")),
            ("Gross Profit (TTM)", fd.get("grossProfits")),
            ("EBITDA", fd.get("ebitda")),
            ("Net Income (Common)", ks.get("netIncomeToCommon")),
            ("Profit Margin", fd.get("profitMargins") or ks.get("profitMargins")),
            ("Operating Margin", fd.get("operatingMargins")),
            ("Gross Margin", fd.get("grossMargins")),
            ("Return on Equity", fd.get("returnOnEquity")),
            ("Return on Assets", fd.get("returnOnAssets")),
            ("Earnings Growth", fd.get("earningsGrowth")),
            ("Total Cash", fd.get("totalCash")),
            ("Total Cash per Share", fd.get("totalCashPerShare")),
            ("Total Debt", fd.get("totalDebt")),
            ("Debt to Equity", fd.get("debtToEquity")),
            ("Current Ratio", fd.get("currentRatio")),
            ("Quick Ratio", fd.get("quickRatio")),
            ("Book Value per Share", ks.get("bookValue")),
            ("Free Cash Flow", fd.get("freeCashflow")),
            ("Operating Cash Flow", fd.get("operatingCashflow")),
            ("Shares Outstanding", ks.get("sharesOutstanding")),
            ("Float Shares", ks.get("floatShares")),
            ("Held by Insiders", ks.get("heldPercentInsiders")),
            ("Held by Institutions", ks.get("heldPercentInstitutions")),
        ]

        lines = [f"{label}: {value}" for label, value in fields if value is not None]
        if not lines:
            return f"No fundamentals data found for symbol '{ticker}'"

        header = (
            f"# Company Fundamentals for {ticker_u} (yahooquery)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + "\n".join(lines)

    except Exception as e:  # noqa: BLE001
        return f"Error retrieving fundamentals for {ticker}: {e}"


def _statement_csv(
    ticker: str,
    freq: str,
    method_name: str,
    label: str,
    curr_date: str | None,
) -> str:
    try:
        ticker_u = ticker.upper()
        t = Ticker(ticker_u)
        frequency = "q" if str(freq).lower().startswith("q") else "a"
        df = _retry(lambda: getattr(t, method_name)(frequency=frequency))
        if df is None or (isinstance(df, str)) or (hasattr(df, "empty") and df.empty):
            return f"No {label.lower()} data found for symbol '{ticker}'"
        wide = _pivot_to_wide(df)
        wide = _filter_by_curr_date(wide, curr_date)
        if wide is None or wide.empty:
            return f"No {label.lower()} data found for symbol '{ticker}'"
        header = (
            f"# {label} data for {ticker_u} ({frequency}, yahooquery)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + wide.to_csv()
    except Exception as e:  # noqa: BLE001
        return f"Error retrieving {label.lower()} for {ticker}: {e}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    return _statement_csv(ticker, freq, "balance_sheet", "Balance Sheet", curr_date)


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    return _statement_csv(ticker, freq, "cash_flow", "Cash Flow", curr_date)


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    return _statement_csv(ticker, freq, "income_statement", "Income Statement", curr_date)
