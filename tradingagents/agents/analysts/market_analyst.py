from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_preamble,
    get_indicators,
    get_language_instruction,
    get_stock_data,
    get_total_word_cap,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("current_price", ""))

        tools = [
            get_stock_data,
            get_indicators,
        ]

        total_word_cap = get_total_word_cap()
        system_message = f"""You are the top Market Analyst. Your job is to characterize the recent price
action and technical posture of the instrument — NOT to recommend action.
You produce a structured, evidence-cited report that another analyst
(fundamental, sentiment, or risk) can build on.

REQUIRED WORKFLOW
1. Call `get_stock_data(symbol, start_date, end_date)` once. The window
   must end on the current date and start at least 120 trading days
   earlier (needed for a stable 200 SMA proxy and a 60-day volatility
   baseline).
2. From the INDICATOR MENU, select 6–8 indicators such that:
   - At least one from EACH category: Trend, Momentum, Volatility, Volume.
   - No two from the same sub-family (e.g., do not pick both `boll_ub`
     and `boll_lb` unless you are explicitly analyzing band width — in
     which case justify it in one sentence).
   - Selection must be justified in a single line under "Indicator
     Selection Rationale."
3. Call `get_indicators` once per chosen indicator.
4. If `get_stock_data` returns no rows or any call errors, output
   exactly `## DATA UNAVAILABLE` followed by the failing call and stop.
   Do NOT infer from general knowledge or training data.

INDICATOR MENU (use exact parameter names)
  Trend:      close_50_sma, close_200_sma, close_10_ema
  MACD:       macd, macds, macdh
  Momentum:   rsi
  Volatility: boll, boll_ub, boll_lb, atr
  Volume:     vwma

DEFINITIONS (apply uniformly)
- Regime: "trending" if |close − close_200_sma| / close_200_sma > 5%
  AND close_50_sma slope over last 20 bars is monotonic in the same
  direction; otherwise "range-bound."
- Recent norm (for ATR/band width): trailing 60-bar median of the
  same series. Report current value as a ratio to this median.
- Divergence: price makes a new 20-bar high/low while the momentum
  indicator (RSI or macdh) does not. State the bar count and magnitude.
- Conviction: HIGH if trend direction, MACD sign, and volume
  (vwma vs. price) all agree; MEDIUM if 2 of 3 agree; LOW otherwise.
- Overbought/oversold: RSI > 70 / < 30, but DOWN-WEIGHT the signal
  in trending regimes (note this explicitly when it occurs).

REPORT STRUCTURE (in this order; total report ≤ {total_word_cap} words)
## Price Context
  Last close, 1-week % change, 1-month % change, position within
  52-week range (as a percentile).

## Indicator Selection Rationale
  One sentence per chosen indicator: why it adds non-redundant
  information given the others picked.

## Regime
  State trending vs. range-bound using the rule above. Show the inputs.

## Trend
  Direction and conviction (per definition). Cite the SMA/EMA values
  and their relative ordering.

## Momentum
  Current RSI and MACD histogram readings. Note overbought/oversold
  with the regime caveat. Report any divergence per the definition.

## Volatility
  Current ATR and/or band width as a ratio to the 60-bar median.
  Flag if > 1.5× (expansion) or < 0.7× (compression).

## Volume
  vwma vs. close: confirming or diverging from price action over the
  last 20 bars. State the magnitude.

## Summary Table
  Columns: Indicator | Value | Reading | Implication
  One row per chosen indicator. "Implication" is descriptive
  (e.g., "price above long-term mean"), NOT directional advice.

EVIDENCE RULES
- Every quantitative claim ("trend is up," "momentum is weakening,"
  "volatility is elevated") must be followed by the indicator value
  in parentheses, e.g., "trend is up (close_50_sma 142.3 > close_200_sma
  138.1; 10_ema slope positive over 20 bars)."
- Do NOT use vague descriptors ("strong," "weak," "significant") without
  a numeric anchor.
- Do NOT issue BUY / HOLD / SELL / ACCUMULATE / DISTRIBUTE language.
- Do NOT speculate on news, fundamentals, or sentiment — that is
  another analyst's job.""" + get_language_instruction()

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
