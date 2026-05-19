"""Tests for IDX lot/tick/ARA-ARB rules.

Pure-Python logic; no external dependencies.
"""

import pytest

from tradingagents.dataflows.idx_trading_rules import (
    IDX_LOT,
    adjust_proposal_for_idx,
    ara_arb_levels,
    get_band,
    is_idx_ticker,
    round_to_lot,
    snap_to_tick,
    trader_idx_footer,
)


@pytest.mark.unit
class TestGetBand:
    def test_band_boundaries_are_half_open(self):
        # exactly 200 belongs to the [200,500) band, not [0,200).
        assert get_band(199).tick == 1
        assert get_band(200).tick == 2
        assert get_band(500).tick == 5
        assert get_band(2000).tick == 10
        assert get_band(5000).tick == 25

    def test_high_price_lands_in_top_band(self):
        assert get_band(50_000).tick == 25

    def test_zero_or_negative_raises(self):
        with pytest.raises(ValueError):
            get_band(0)
        with pytest.raises(ValueError):
            get_band(-100)


@pytest.mark.unit
class TestSnapToTick:
    def test_nearest_rounds_to_legal_tick(self):
        # Rp 1234 in the [500,2000) band, tick 5 → nearest is 1235.
        assert snap_to_tick(1234) == 1235

    def test_down_floors_in_band(self):
        # 2,001 in [2000,5000), tick 10 → floor = 2000.
        assert snap_to_tick(2001, mode="down") == 2000

    def test_up_ceilings_in_band(self):
        # 2,001 in [2000,5000), tick 10 → ceil = 2010.
        assert snap_to_tick(2001, mode="up") == 2010

    def test_high_price_snaps_to_25(self):
        assert snap_to_tick(7_512) == 7_500  # nearest 25
        assert snap_to_tick(7_512, mode="up") == 7_525

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            snap_to_tick(1000, mode="sideways")


@pytest.mark.unit
class TestRoundToLot:
    def test_floor_default(self):
        assert round_to_lot(350) == 300
        assert round_to_lot(99) == 0

    def test_up(self):
        assert round_to_lot(101, mode="up") == 200
        assert round_to_lot(IDX_LOT, mode="up") == IDX_LOT

    def test_nearest(self):
        assert round_to_lot(149, mode="nearest") == 100
        assert round_to_lot(150, mode="nearest") == 200

    def test_non_positive_is_zero(self):
        assert round_to_lot(0) == 0
        assert round_to_lot(-5) == 0


@pytest.mark.unit
class TestAraArbLevels:
    def test_25_pct_band_around_mid_price(self):
        ara, arb = ara_arb_levels(1_000)
        # [500,2000): tick 5, ARA/ARB 25%.
        assert ara == 1_250  # 1000 * 1.25 = 1250, snapped down
        assert arb == 750    # 1000 * 0.75 = 750, snapped up

    def test_20_pct_band_above_5k(self):
        ara, arb = ara_arb_levels(10_000)
        assert ara == 12_000  # 10000*1.2, tick 25
        assert arb == 8_000

    def test_results_are_on_legal_ticks(self):
        ara, arb = ara_arb_levels(347)  # [200,500), tick 2
        assert ara % 2 == 0
        assert arb % 2 == 0


@pytest.mark.unit
class TestIsIdxTicker:
    def test_jk_suffix_detected(self):
        assert is_idx_ticker("BBCA.JK")
        assert is_idx_ticker("bbca.jk")

    def test_bare_or_other_suffix_not_idx(self):
        assert not is_idx_ticker("BBCA")
        assert not is_idx_ticker("AAPL")
        assert not is_idx_ticker("9988.HK")


class _DummyProposal:
    """Lightweight stand-in for TraderProposal: no Pydantic dependency in the test."""

    def __init__(self, entry_price=None, stop_loss=None):
        self.entry_price = entry_price
        self.stop_loss = stop_loss

    def model_copy(self, update):
        new = _DummyProposal(self.entry_price, self.stop_loss)
        for k, v in update.items():
            setattr(new, k, v)
        return new


@pytest.mark.unit
class TestAdjustProposalForIdx:
    def test_snaps_entry_and_stop(self):
        p = _DummyProposal(entry_price=1_234, stop_loss=987)
        out = adjust_proposal_for_idx(p)
        assert out.entry_price == 1_235  # tick 5, nearest
        assert out.stop_loss == 985

    def test_leaves_missing_fields_alone(self):
        p = _DummyProposal(entry_price=None, stop_loss=None)
        out = adjust_proposal_for_idx(p)
        assert out is p  # no copy made when there's nothing to update

    def test_ignores_non_positive(self):
        p = _DummyProposal(entry_price=0, stop_loss=-5)
        out = adjust_proposal_for_idx(p)
        assert out.entry_price == 0
        assert out.stop_loss == -5


@pytest.mark.unit
class TestTraderIdxFooter:
    def test_empty_on_blank_price(self):
        assert trader_idx_footer("") == ""
        assert trader_idx_footer(None) == ""

    def test_includes_rules_block(self):
        out = trader_idx_footer("Rp 4,500")
        assert "IDX Trading Rules" in out
        assert "tick = Rp10" in out  # 4500 in [2000,5000) band
        assert "lot = 100" in out

    def test_parses_numeric_strings(self):
        # The price anchor is typically formatted with commas + currency.
        out = trader_idx_footer("8,275.00 IDR")
        assert "IDX Trading Rules" in out
