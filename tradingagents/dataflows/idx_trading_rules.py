"""IDX (Indonesia Stock Exchange) lot, tick, and auto-rejection rules.

Indonesia's BEI trades in lots of 100 shares with a stepped tick
schedule (fraksi harga) and daily price bands (ARA / ARB). This module
is the single source of truth for those constants.

Reference: SE-00118/BEI/12-2022 (tick schedule, eff. Dec 2022) and the
post-2023 symmetric auto-rejection regime. If BEI updates the rule
again, edit ``TICK_BANDS`` here.

Pure-Python, no I/O: safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

IDX_LOT = 100  # shares per lot


@dataclass(frozen=True)
class TickBand:
    """A half-open price band [lower, upper) with its tick size and ARA/ARB."""
    lower: float           # inclusive lower bound
    upper: float           # exclusive upper bound (use float('inf') for top band)
    tick: int              # tick increment in IDR
    ara: float             # auto-rejection-atas as a fraction (0.25 = 25%)
    arb: float             # auto-rejection-bawah as a fraction


# Bands ordered low → high. ARA / ARB reflect the post-2023 symmetric
# regime. If BEI re-tightens or relaxes, edit here.
TICK_BANDS: tuple[TickBand, ...] = (
    TickBand(lower=0,     upper=200,           tick=1,  ara=0.35, arb=0.35),
    TickBand(lower=200,   upper=500,           tick=2,  ara=0.25, arb=0.25),
    TickBand(lower=500,   upper=2_000,         tick=5,  ara=0.25, arb=0.25),
    TickBand(lower=2_000, upper=5_000,         tick=10, ara=0.20, arb=0.20),
    TickBand(lower=5_000, upper=float("inf"),  tick=25, ara=0.20, arb=0.20),
)


def get_band(price: float) -> TickBand:
    """Return the TickBand containing ``price``. Raises ValueError for non-positive."""
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    for band in TICK_BANDS:
        if band.lower <= price < band.upper:
            return band
    return TICK_BANDS[-1]  # safety net (price = inf shouldn't happen)


def snap_to_tick(price: float, mode: str = "nearest") -> int:
    """Snap a rupiah price to the legal tick for its band.

    mode: ``"nearest"`` (default), ``"up"`` (ceiling), or ``"down"`` (floor).
    Returns int rupiah.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    tick = get_band(price).tick
    q = price / tick
    if mode == "up":
        snapped = ceil(q)
    elif mode == "down":
        snapped = floor(q)
    elif mode == "nearest":
        snapped = int(round(q))
    else:
        raise ValueError(f"unknown mode {mode!r}; want nearest/up/down")
    return int(snapped * tick)


def round_to_lot(shares: float, mode: str = "down") -> int:
    """Round a share count to whole lots of 100. ``mode``: down/up/nearest."""
    if shares <= 0:
        return 0
    lots = shares / IDX_LOT
    if mode == "up":
        whole = ceil(lots)
    elif mode == "nearest":
        whole = int(round(lots))
    elif mode == "down":
        whole = floor(lots)
    else:
        raise ValueError(f"unknown mode {mode!r}; want nearest/up/down")
    return int(whole * IDX_LOT)


def ara_arb_levels(reference_price: float) -> tuple[int, int]:
    """Return ``(ARA, ARB)`` — next-session ceiling and floor in IDR,
    derived from ``reference_price`` (typically prior close).
    Both snapped to the legal tick: ARA rounded down (a tradeable
    ceiling), ARB rounded up (a tradeable floor).
    """
    band = get_band(reference_price)
    return (
        snap_to_tick(reference_price * (1 + band.ara), mode="down"),
        snap_to_tick(reference_price * (1 - band.arb), mode="up"),
    )


def format_rules_block(reference_price: float) -> str:
    """One-line human-readable rule summary for prompts and reports."""
    band = get_band(reference_price)
    ara, arb = ara_arb_levels(reference_price)
    return (
        f"IDX execution rules at reference price Rp{int(reference_price):,}: "
        f"tick = Rp{band.tick}; lot = {IDX_LOT} shares; "
        f"next-session ARA ceiling ≈ Rp{ara:,} (+{band.ara*100:.0f}%), "
        f"ARB floor ≈ Rp{arb:,} (-{band.arb*100:.0f}%). "
        "Limit prices must be exact multiples of the tick; "
        "share quantities must be whole lots."
    )


def is_idx_ticker(ticker: str) -> bool:
    """Heuristic: True for explicit ``.JK`` suffix."""
    return bool(ticker) and ticker.upper().endswith(".JK")


# ---------------------------------------------------------------------------
# Trader proposal post-processing
# ---------------------------------------------------------------------------


def _parse_price(raw) -> float | None:
    """Pull the first positive numeric value out of a free-form price string.

    Handles common formats: ``"4500"``, ``"4,500.00 IDR"``, ``"Rp 4,500"``.
    Returns ``None`` if no positive number can be extracted.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    cleaned = str(raw).replace(",", "")
    for token in cleaned.split():
        try:
            val = float(token)
            if val > 0:
                return val
        except ValueError:
            continue
    return None


def adjust_proposal_for_idx(proposal):
    """Return a copy of a TraderProposal with entry/stop snapped to legal IDX ticks.

    Defined to accept the Pydantic model duck-typed (entry_price / stop_loss
    attributes) so this module doesn't import the schemas.
    """
    if proposal is None:
        return proposal
    updates: dict[str, float] = {}
    entry = getattr(proposal, "entry_price", None)
    stop = getattr(proposal, "stop_loss", None)
    if isinstance(entry, (int, float)) and entry > 0:
        updates["entry_price"] = float(snap_to_tick(entry, mode="nearest"))
    if isinstance(stop, (int, float)) and stop > 0:
        updates["stop_loss"] = float(snap_to_tick(stop, mode="nearest"))
    if not updates:
        return proposal
    if hasattr(proposal, "model_copy"):
        return proposal.model_copy(update=updates)
    return proposal


def trader_idx_footer(current_price) -> str:
    """Return a markdown footer with the active IDX rules block,
    or ``""`` if the price can't be parsed.
    """
    price = _parse_price(current_price)
    if price is None:
        return ""
    return "\n\n---\n\n### IDX Trading Rules\n" + format_rules_block(price)
