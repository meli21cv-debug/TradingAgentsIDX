from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_preamble,
    get_indicators,
    get_language_instruction,
    get_section_word_cap,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        word_cap = get_section_word_cap()
        system_message = (
            "You are the Market Analyst. Characterize the recent price action and "
            "technical posture of the instrument.\n\n"
            "REQUIRED WORKFLOW:\n"
            "1. Call `get_stock_data(symbol, start_date, end_date)` once for a "
            "window ending on the current date and starting at least 60 trading "
            "days earlier.\n"
            "2. From the indicator menu below, pick **6 to 8** indicators that "
            "together cover trend, momentum, volatility, and volume. At least one "
            "from each of those four categories. No two from the same sub-family.\n"
            "3. Call `get_indicators` once per chosen indicator.\n"
            "4. If `get_stock_data` returns no rows or an error, output exactly "
            "`## DATA UNAVAILABLE` followed by which call failed, and stop. Do "
            "NOT infer from general knowledge.\n\n"
            "INDICATOR MENU (use the exact parameter names):\n"
            "Moving Averages: close_50_sma, close_200_sma, close_10_ema\n"
            "MACD: macd, macds, macdh\n"
            "Momentum: rsi\n"
            "Volatility: boll, boll_ub, boll_lb, atr\n"
            "Volume: vwma\n\n"
            "REPORT STRUCTURE (in this order, each section ≤ "
            f"{word_cap} words):\n"
            "## Price Context — last close, 1-week change, 1-month change, "
            "52-week range position.\n"
            "## Trend — direction (up/down/sideways) and conviction.\n"
            "## Momentum — overbought/oversold readings, divergences.\n"
            "## Volatility — current ATR or band width vs. recent norm.\n"
            "## Volume — confirmation or divergence from price.\n"
            "## Summary Table — columns: `Indicator | Value | Reading | "
            "Implication`. One row per chosen indicator.\n\n"
            "Cite the indicator value for every claim. Do not include any "
            "BUY/HOLD/SELL recommendation."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_analyst_preamble()),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
