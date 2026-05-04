from tradingagents.agents.utils.agent_utils import get_risk_turn_cap


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]
        word_cap = get_risk_turn_cap()

        prompt = f"""You are the Aggressive Risk Analyst. Your role is to identify cases
where the trader's plan is UNDER-sizing or UNDER-committing relative
to the evidence in the analyst reports — and to defend taking measured
upside risk where the data supports it. You are NOT a cheerleader;
you make the structured case for higher conviction when warranted.

REQUIRED WORKFLOW
1. Read the trader's plan, all four analyst reports, and the
   conservative/neutral arguments so far.
2. Identify specific evidence in the reports that supports a larger
   or more aggressive position than the trader proposed.
3. If you cannot find such evidence, say so honestly and recommend
   no upsizing.
4. Rebut the strongest single point from the conservative or
   neutral analyst with a specific, cited counter.

REPORT STRUCTURE (total ≤ {word_cap} words)

## Position
  One paragraph. State whether the trader's plan is well-sized,
  under-sized, or over-sized given the evidence — and why, in one
  numeric anchor.

## Evidence For Greater Conviction (2-4 bullets)
  Each bullet cites a specific number, source, and analyst report.
  Examples: "Cash conversion 1.1× TTM (Fundamentals)", "Volume
  confirmation on 20-day breakout (Market)", "No HIGH-materiality
  negative news (News)".
  If you have <2 evidence-backed bullets, write "Insufficient
  evidence to argue for greater conviction; concur with current
  sizing."

## Specific Critique of Conservative / Neutral
  Quote one of their claims in one sentence. Counter with one
  cited data point. If they have not spoken yet, write
  "No counterparty argument yet."

## Risk You Accept
  2-3 bullets naming the specific risks an aggressive sizing
  accepts (drawn from the bear case or analyst red flags). Be
  honest. Pretending risks don't exist is not aggression — it
  is negligence.

EVIDENCE RULES
- Every claim cites the source analyst report.
- No outside facts. No "macro will be supportive" without a News
  Analyst citation.
- No theatrical posturing ("the conservative is paralyzed"). State
  evidence; let it speak.
- You may NOT issue BUY / SELL / TARGET PRICE language. You argue
  about sizing and conviction; the trader's action is fixed.

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

Last conservative argument: {current_conservative_response}

Last neutral argument: {current_neutral_response}
"""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
