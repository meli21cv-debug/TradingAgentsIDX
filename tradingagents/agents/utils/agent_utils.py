from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


_DEPTH_TO_SECTION_WORDS = {
    1: 100,   # Shallow
    3: 200,   # Medium
    5: 350,   # Deep
}

_DEPTH_TO_TOTAL_WORDS = {
    1: 800,    # Shallow
    3: 1500,   # Medium
    5: 2500,   # Deep
}

_DEPTH_TO_NEWS_LOOKBACK_DAYS = {
    1: 14,    # Shallow
    3: 30,    # Medium
    5: 60,    # Deep
}

# Per-turn caps for debaters. Kept small so debate volume does not exceed
# the source material from analysts.
_DEPTH_TO_RESEARCHER_TURN = {
    1: 200,   # Shallow
    3: 350,   # Medium
    5: 500,   # Deep
}

_DEPTH_TO_RISK_TURN = {
    1: 150,   # Shallow
    3: 250,   # Medium
    5: 350,   # Deep
}


def _depth_tier() -> int:
    from tradingagents.dataflows.config import get_config
    depth = int(get_config().get("max_debate_rounds", 1))
    if depth <= 1:
        return 1
    if depth <= 3:
        return 3
    return 5


def get_section_word_cap() -> int:
    """Per-section word cap for analyst reports, scaled by research depth."""
    return _DEPTH_TO_SECTION_WORDS[_depth_tier()]


def get_total_word_cap() -> int:
    """Total report word cap, scaled by research depth."""
    return _DEPTH_TO_TOTAL_WORDS[_depth_tier()]


def get_news_lookback_days() -> int:
    """News-window lookback in days for News and Social analysts, by depth."""
    return _DEPTH_TO_NEWS_LOOKBACK_DAYS[_depth_tier()]


def get_researcher_turn_cap() -> int:
    """Per-turn word cap for Bull/Bear researchers — kept tight to prevent
    debate volume from dwarfing analyst evidence."""
    return _DEPTH_TO_RESEARCHER_TURN[_depth_tier()]


def get_risk_turn_cap() -> int:
    """Per-turn word cap for risk debaters (Aggressive/Conservative/Neutral).
    Tighter than researchers because there are three of them."""
    return _DEPTH_TO_RISK_TURN[_depth_tier()]


_ANALYST_PREAMBLE = (
    "You are a helpful AI assistant collaborating with other analysts. Use the "
    "provided tools to answer the question. If you cannot fully answer, that is "
    "OK — another analyst will pick up where you left off. Execute what you can.\n"
    "You have access to the following tools: {tool_names}.\n"
    "Do NOT include any FINAL TRANSACTION PROPOSAL or BUY/HOLD/SELL recommendation "
    "— that decision belongs to the Trader downstream.\n"
    "{system_message}\n"
    "Current date: {current_date}. {instrument_context}"
)


def get_analyst_preamble() -> str:
    """Shared system-prompt preamble for all analyst agents."""
    return _ANALYST_PREAMBLE


def build_instrument_context(ticker: str, current_price: str = "") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers
    and share the same current-price anchor."""
    from tradingagents.dataflows.config import get_config
    base = (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `.JK`)."
    )
    if current_price:
        base += (
            f" Latest close: {current_price}. Use this as the current-price "
            "anchor when discussing entry/stop/target zones, % moves, and "
            "valuation; do not assume a different price."
        )
    if (get_config().get("market") or "").upper() == "ID":
        base += (
            " This is an Indonesian (IDX) stock. When news searches return little, "
            "retry with Bahasa Indonesia keywords (e.g. 'saham', 'IHSG', 'laba bersih', "
            "'dividen', the company's Indonesian name) and consider the bare ticker "
            "without the `.JK` suffix as a search term."
        )
    return base

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
