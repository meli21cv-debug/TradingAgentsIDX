from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.idx_corporate_actions import (
    get_corporate_actions as _get_corporate_actions,
)
from tradingagents.dataflows.market_utils import apply_market_suffix


@tool
def get_corporate_actions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """
    Retrieve recent and upcoming corporate actions for a ticker:
    dividend history, stock splits, upcoming earnings/ex-dividend dates,
    plus Indonesian-news mentions of RUPS, rights issues, buybacks, and
    M&A within ±90 days. Useful for valuation context (dilution, cash
    return) and event-driven risk.

    Args:
        ticker: Ticker symbol of the company.
        curr_date: Current trading date in yyyy-mm-dd (used to centre the
            news scan window).
    Returns:
        Formatted markdown summary of corporate actions.
    """
    return _get_corporate_actions(apply_market_suffix(ticker), curr_date)
