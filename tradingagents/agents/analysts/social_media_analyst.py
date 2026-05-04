from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_preamble,
    get_language_instruction,
    get_news,
    get_section_word_cap,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
        ]

        word_cap = get_section_word_cap()
        system_message = (
            "You are the Social Sentiment Analyst. Characterize what retail and "
            "trading commentators are saying about the instrument over the past "
            "7 days, using `get_news` as the proxy source (it aggregates "
            "retail-investor outlets).\n\n"
            "REQUIRED WORKFLOW:\n"
            "1. Call `get_news(ticker, start_date, end_date)` with start_date = "
            "current_date minus 7 days.\n"
            "2. If it returns only 'No news found' or errors, output "
            "`## DATA UNAVAILABLE` and stop.\n\n"
            f"REPORT STRUCTURE (each section ≤ {word_cap} words):\n"
            "## Sentiment Direction — bullish / bearish / mixed / neutral, in "
            "one sentence, with a count of stories supporting each side.\n"
            "## Recurring Themes — 3-5 bullets of repeated talking points "
            "(e.g. 'buyback announcement', 'foreign outflow'). Quote a 5-10 "
            "word excerpt for each.\n"
            "## Notable Sources — top 3 outlets by article count.\n"
            "## Retail Concerns / Excitements — 2-4 specific worries or hopes "
            "expressed in the coverage.\n"
            "## Summary Table — columns: `Theme | Stance | Article Count | "
            "Representative Source`.\n\n"
            "Cite the source for every theme. Do not include any "
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
