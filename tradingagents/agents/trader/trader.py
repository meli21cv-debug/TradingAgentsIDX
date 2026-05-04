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


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name, state.get("current_price", ""))
        investment_plan = state["investment_plan"]

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

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
                    "4. Identify the single most material counterargument from "
                    "the Bear (or Bull, if rating is bearish) and state the "
                    "specific data point that would invalidate the trade.\n"
                    "5. If the Research Manager's rating is Hold OR any analyst "
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
                    "- Invalidation conditions: specific, observable.\n\n"
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

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
