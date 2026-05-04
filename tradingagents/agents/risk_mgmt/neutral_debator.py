from tradingagents.agents.utils.agent_utils import get_total_word_cap


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]
        word_cap = get_total_word_cap() // 3

        prompt = f"""You are the Neutral Risk Analyst. Your role is NOT to split the
difference between aggressive and conservative — that is laziness
disguised as balance. Your role is to identify where each side is
right and where each side is wrong on the evidence, and to recommend
the sizing/risk-control combination that the data actually supports,
even if it lands at one extreme.

REQUIRED WORKFLOW
1. Read the trader's plan, all four analyst reports, and the
   aggressive/conservative arguments.
2. For each side's strongest claim, judge whether the cited evidence
   actually supports it (or whether it cherry-picks).
3. Recommend a sizing/risk-control combination grounded in the
   reports. If that combination matches the aggressive side, say so.
   If it matches the conservative side, say so. Do not auto-average.

REPORT STRUCTURE (total ≤ {word_cap} words)

## Position
  One paragraph. State your recommendation on sizing and risk
  controls relative to the trader's plan, with the single most
  important data point that drives it.

## Where Each Side Is Right
  - Aggressive: 1-2 cited claims from their argument that survive
    scrutiny against the analyst reports.
  - Conservative: 1-2 cited claims from their argument that survive
    scrutiny against the analyst reports.
  If either side has not spoken yet, write "Has not spoken."

## Where Each Side Is Wrong
  - Aggressive: 1-2 cited claims that misread or cherry-pick the
    reports. State the contradiction with a specific number.
  - Conservative: 1-2 cited claims that misread or cherry-pick the
    reports. State the contradiction with a specific number.

## Recommended Adjustment
  The specific sizing and risk-control changes to the trader's plan,
  each tied to a cited indicator value or metric. May match
  aggressive, conservative, or neither — what the data supports.

EVIDENCE RULES
- Every claim cites the source analyst report.
- No outside facts. No "history shows that...".
- No theatrical posturing. State evidence; let it speak.
- You may NOT issue BUY / SELL / TARGET PRICE language.
- Balance is not the goal. Calibration is. Lopsided evidence
  warrants a lopsided recommendation.

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

Last conservative argument: {current_conservative_response}
"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
