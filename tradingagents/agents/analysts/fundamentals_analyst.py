from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_preamble,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
    get_section_word_cap,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        word_cap = get_section_word_cap()
        system_message = (
            "You are the Fundamentals Analyst. Summarize the company's recent "
            "financial health.\n\n"
            "REQUIRED WORKFLOW:\n"
            "1. Call `get_fundamentals(ticker, curr_date)`.\n"
            "2. Call `get_income_statement`, `get_balance_sheet`, and "
            "`get_cashflow` (each with `freq=\"quarterly\"`).\n"
            "3. If `get_fundamentals` errors AND at least two of the statement "
            "calls also fail, output `## DATA UNAVAILABLE` and stop.\n\n"
            f"REPORT STRUCTURE (each section ≤ {word_cap} words):\n"
            "## Profile — sector, business model, market cap.\n"
            "## Profitability — revenue and net income trend over the last 4 "
            "quarters; margin direction.\n"
            "## Balance Sheet — debt level, cash position, key changes vs. "
            "prior quarter.\n"
            "## Cash Flow — operating CF trend, free cash flow, capex direction.\n"
            "## Valuation — P/E, P/B, dividend yield as available. Compare to "
            "sector if data permits.\n"
            "## Red Flags / Strengths — 2-4 bullets each (or 'None apparent').\n"
            "## Summary Table — columns: `Metric | Latest | Prior Quarter | "
            "YoY Change | Verdict`.\n\n"
            "Cite the line item or statement for every figure. Do not include "
            "any BUY/HOLD/SELL recommendation."
            + get_language_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
