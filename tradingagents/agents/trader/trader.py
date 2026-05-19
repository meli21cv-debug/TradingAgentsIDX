"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.idx_trading_rules import (
    _parse_price,
    adjust_proposal_for_idx,
    format_rules_block,
    trader_idx_footer,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        current_price = state.get("current_price", "")
        instrument_context = build_instrument_context(company_name, current_price)
        investment_plan = state["investment_plan"]

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        # IDX execution-rules block: when the active market is Indonesia,
        # the Trader's entry/stop must land on legal ticks and sizing
        # must be in whole lots of 100. We inject the rules into the
        # prompt and snap prices post-hoc as defence-in-depth.
        is_idx = (get_config().get("market") or "").upper() == "ID"
        idx_rules_prompt = ""
        if is_idx:
            ref_price = _parse_price(current_price)
            if ref_price is not None:
                idx_rules_prompt = (
                    "\n\nIDX EXECUTION CONSTRAINTS (mandatory)\n"
                    f"- {format_rules_block(ref_price)}\n"
                    "- Quote `entry_price` and `stop_loss` as integer rupiah "
                    "values on the legal tick. Do NOT use fractional rupiah.\n"
                    "- When describing position size, state share count as "
                    "whole lots of 100 (e.g. '5 lots = 500 shares'), not "
                    "individual shares.\n"
                    "- A target above the ARA ceiling or stop below the ARB "
                    "floor cannot fill in a single session — call out the "
                    "multi-day path explicitly if your levels exceed the "
                    "next-session bands."
                )
            else:
                idx_rules_prompt = (
                    "\n\nIDX EXECUTION CONSTRAINTS (mandatory)\n"
                    "- Quote `entry_price` and `stop_loss` as integer rupiah "
                    "values on legal IDX ticks (Rp1 below Rp200, Rp2 in "
                    "Rp200-500, Rp5 in Rp500-2000, Rp10 in Rp2000-5000, "
                    "Rp25 above Rp5000).\n"
                    "- State position size in whole lots of 100 shares."
                )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Trader. Translate the Research Manager's rating "
                    "into a concrete, sized, time-bounded transaction proposal. "
                    "You are accountable for execution risk: entry zone, exit "
                    "rules, position size, and what would force you to abandon "
                    "the trade. Your reasoning must trace from specific numbers "
                    "in the analyst reports — never from feel or vague intent."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{instrument_context}\n\n"
                    "REQUIRED WORKFLOW\n"
                    "1. Take the Research Manager's rating as the directional anchor.\n"
                    "2. Pull entry/exit levels from the Market Analyst's "
                    "indicator values (price context, ATR, band positions, SMA "
                    "levels). Do NOT invent levels not supported by the report.\n"
                    "3. Size the position relative to volatility — quote ATR "
                    "(or band width) from the Market Analyst as the volatility "
                    "anchor.\n"
                    "4. **Fair-value gap check (mandatory)**: Read the "
                    "Fundamentals analyst's Fair Value Estimate section. State "
                    "the gap between the current market price (from the "
                    "instrument context) and the fair-value mid-point as a "
                    "percentage. A larger discount (for a bullish rating) or "
                    "premium (for a bearish rating) warrants higher conviction "
                    "and possibly larger size — within the Research Manager's "
                    "rating tier. A gap that contradicts the rating direction "
                    "(e.g., Buy rating but market trades 20% above fair value) "
                    "must be acknowledged explicitly and either downgrade "
                    "conviction or articulate why it doesn't matter (e.g., "
                    "thesis is short-term technical, not valuation).\n"
                    "5. Identify the single most material counterargument from "
                    "the Bear (or Bull, if rating is bearish) and state the "
                    "specific data point that would invalidate the trade.\n"
                    "6. If the Research Manager's rating is Hold OR any analyst "
                    "report returned DATA UNAVAILABLE, default to a 'no action' "
                    "or a small probe-size position with explicit reasoning.\n\n"
                    "EVIDENCE RULES\n"
                    "- Every entry, stop, target, and size must be tied to a "
                    "specific number from the Market Analyst report or to the "
                    "Research Manager's rating threshold.\n"
                    "- Do not invent price targets that aren't grounded in "
                    "indicator values or sources cited by the analysts.\n"
                    "- Do not contradict the Research Manager's rating without "
                    "stating why explicitly and citing the data that overrides.\n\n"
                    "OUTPUT EXPECTATIONS\n"
                    "- Action: BUY / HOLD / SELL (and ONLY here, in the trader "
                    "output, is this language permitted).\n"
                    "- Conviction: stated explicitly (low / medium / high) and "
                    "justified.\n"
                    "- Sizing rationale: tied to volatility metric (ATR or band "
                    "width).\n"
                    "- Entry zone, stop level, target level: each tied to a "
                    "cited indicator value.\n"
                    "- Time horizon and review trigger.\n"
                    "- Invalidation conditions: specific, observable."
                    f"{idx_rules_prompt}\n\n"
                    "---\n\n"
                    f"Research Manager's investment plan:\n{investment_plan}\n\n"
                    "---\n\n"
                    "Analyst inputs (source of truth for all numbers):\n\n"
                    f"Market Analyst:\n{market_report}\n\n"
                    f"News:\n{news_report}\n\n"
                    f"Social Sentiment:\n{sentiment_report}\n\n"
                    f"Fundamentals:\n{fundamentals_report}\n"
                ),
            },
        ]

        def _render(proposal):
            # On IDX, snap quoted prices to legal ticks before rendering and
            # append the rules footer so downstream agents and saved reports
            # carry the same constraint context.
            if is_idx:
                proposal = adjust_proposal_for_idx(proposal)
            rendered = render_trader_proposal(proposal)
            if is_idx:
                footer = trader_idx_footer(current_price)
                if footer:
                    rendered += footer
            return rendered

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            _render,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
