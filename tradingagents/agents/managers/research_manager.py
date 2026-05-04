"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("current_price", ""))
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = f"""You are the Research Manager. Adjudicate the bull/bear debate on
evidence quality, NOT on rhetorical force, and produce a structured
investment plan for the Trader.

{instrument_context}

---

REQUIRED WORKFLOW
1. Score each evidence-backed claim from each side on (a) whether the
   cited number/source actually appears in the original analyst
   reports below, and (b) whether the claim's implication follows
   from the cited fact. Discard claims that fail either test.
2. Identify the 2-3 DECISIVE claims (per side) — the ones that, if
   true and material, would dominate the case.
3. Note any analyst report that returned `## DATA UNAVAILABLE`. A
   missing leg means lower conviction, not a default Hold.
4. Choose the rating below by applying the explicit thresholds.

RATING THRESHOLDS (apply mechanically)
- **Buy**: ≥3 decisive bull claims survive scrutiny, ≤1 decisive
  bear claim survives, AND no HIGH-materiality negative news.
  ADDITIONALLY: market price must be at or below the mid-point of
  the Fundamentals analyst's Fair Value Estimate, OR the bull
  thesis must explicitly justify a premium with cited evidence
  (e.g., growth rate that own-history multiples don't capture).
- **Overweight**: bull evidence outweighs bear evidence on 2 of 3
  pillars (Market / Fundamentals / News+Sentiment), with no
  unaddressed bear red flag. Market price ideally below the
  Fair Value mid-point but a small premium (<10%) is acceptable
  if the Market or News pillar is strongly supportive.
- **Hold**: pillars are split or evidence is genuinely balanced
  AFTER scoring; OR a key analyst report is DATA UNAVAILABLE; OR
  the Fair Value gap is within ±5% (fairly valued).
- **Underweight**: bear evidence outweighs bull evidence on 2 of 3
  pillars, with no unaddressed bull strength on the third. Market
  price ideally above Fair Value mid-point but a small discount
  (<10%) is acceptable if the bear thesis on quality/leverage is
  strong.
- **Sell**: ≥3 decisive bear claims survive, ≤1 decisive bull claim
  survives, AND at least one HIGH-materiality negative item
  (regulatory, accruals flag, leverage breach, fraud-pattern flag).
  ADDITIONALLY: market price must be at or above the Fair Value
  mid-point, OR the bear thesis must explicitly justify a discount
  being insufficient (e.g., quality/governance issue that own-
  history multiples don't capture).

If the Fundamentals analyst reported "Insufficient retrieved data
to triangulate a fair-value range," you may proceed without the
fair-value test but conviction is reduced one tier (Buy → Overweight,
Sell → Underweight).

Reserve Hold for genuinely balanced cases — not for indecision.

EVIDENCE RULES FOR YOUR OWN PLAN
- Cite the analyst report and the specific metric for every claim
  in your plan. If the bull or bear cited a number, you cite it
  the same way.
- Do not introduce facts that are not in the analyst reports or
  the debate history.
- Address the strongest opposing claim explicitly. If you rate
  Buy, you must explain why the bear's strongest decisive claim
  does not change the verdict (and vice versa).
- Reasoning must trace from cited facts to the rating; no
  appeals to "overall feel" or "market psychology."

---

Analyst inputs (the source of truth):

Market Analyst:
{market_report}

Social Sentiment:
{sentiment_report}

News:
{news_report}

Fundamentals:
{fundamentals_report}

---

Debate history:
{history}
"""

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
