from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_preamble,
    get_global_news,
    get_language_instruction,
    get_news,
    get_section_word_cap,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        word_cap = get_section_word_cap()
        system_message = (
            "You are the News Analyst. Surface news from the past 7 days that "
            "materially affects the instrument or its sector, and characterize "
            "the macro backdrop.\n\n"
            "REQUIRED WORKFLOW:\n"
            "1. Call `get_news(ticker, start_date, end_date)` with start_date = "
            "current_date minus 7 days.\n"
            "2. Call `get_global_news(curr_date)`.\n"
            "3. If both calls return only 'No news found' or errors, output "
            "`## DATA UNAVAILABLE` and stop. If only one fails, note it "
            "explicitly and continue.\n\n"
            f"REPORT STRUCTURE (each section ≤ {word_cap} words):\n"
            "## Company / Instrument News — bullet each material story as: "
            "`**[Date]** — [Headline]. [1-sentence why it matters]. (Source: "
            "[publisher])`. Drop trivial stories. If fewer than 3 material "
            "stories exist, say so explicitly.\n"
            "## Macro Context — 3-5 bullets covering relevant macro/sector "
            "items from the global news.\n"
            "## Catalysts to Watch — 2-4 forward-looking items (earnings, "
            "dividend dates, sector events) explicitly mentioned in retrieved "
            "articles. If none mentioned, say 'None retrieved'.\n"
            "## Summary Table — columns: `Date | Headline | Source | "
            "Sentiment (+/-/neutral) | Materiality (high/med/low)`.\n\n"
            "Cite the publisher for every story. Do not fabricate news. Do "
            "not include any BUY/HOLD/SELL recommendation."
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
            "news_report": report,
        }

    return news_analyst_node
