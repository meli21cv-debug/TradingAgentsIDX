from tradingagents.agents.utils.agent_utils import get_total_word_cap


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]
        word_cap = get_total_word_cap() // 3

        prompt = f"""You are the Conservative Risk Analyst. Your role is to identify cases
where the trader's plan UNDERSTATES tail risk, OVERSIZES relative to
volatility, or relies on weak evidence — and to argue for tighter
risk controls when the data warrants. You are NOT reflexively
cautious; you make the structured case for risk reduction when the
evidence supports it.

REQUIRED WORKFLOW
1. Read the trader's plan, all four analyst reports, and the
   aggressive/neutral arguments so far.
2. Identify specific evidence in the reports that supports tighter
   stops, smaller size, or no action.
3. If you cannot find such evidence, say so honestly and concede
   the trader's sizing.
4. Rebut the strongest single point from the aggressive or neutral
   analyst with a specific, cited counter.

REPORT STRUCTURE (total ≤ {word_cap} words)

## Position
  One paragraph. State whether the trader's plan is appropriately
  sized, oversized, or under-protected given the evidence — and
  why, with one numeric anchor.

## Evidence For Greater Caution (2-4 bullets)
  Each bullet cites a specific number, source, and analyst report.
  Examples: "Net Debt / EBITDA 5.2× and trending up (Fundamentals)",
  "ATR ratio 1.8× recent norm (Market)", "Goodwill 45% of total
  assets (Fundamentals)", "[unverified] tag on highest-impact news
  (News)".
  If you have <2 evidence-backed bullets, write "Insufficient
  evidence to argue for greater caution; concur with current
  sizing."

## Specific Critique of Aggressive / Neutral
  Quote one of their claims in one sentence. Counter with one
  cited data point. If they have not spoken yet, write
  "No counterparty argument yet."

## What You Would Adjust
  2-3 specific, observable adjustments tied to the analyst data
  (e.g., "tighten stop to 1.5× ATR rather than 2× given band-width
  expansion", "halve size until accruals flag is resolved",
  "delay entry until RSI exits overbought regime"). Each must
  reference an indicator value or metric from the reports.

EVIDENCE RULES
- Every claim cites the source analyst report.
- No outside facts. No "markets feel toppy" without a Market
  Analyst citation.
- No theatrical posturing ("the aggressive analyst is reckless").
  State evidence; let it speak.
- You may NOT issue BUY / SELL / TARGET PRICE language. You argue
  about sizing and risk controls; the trader's action is fixed.
- Caution is not virtue. Argue for it where the data supports it,
  concede it where the data does not.

---

Trader's decision:
{trader_decision}

Market Analyst:
{market_research_report}

Social Sentiment:
{sentiment_report}

News:
{news_report}

Fundamentals:
{fundamentals_report}

Risk debate so far:
{history}

Last aggressive argument: {current_aggressive_response}

Last neutral argument: {current_neutral_response}
"""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
