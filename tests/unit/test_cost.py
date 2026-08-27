"""factory_floor/cost.py (phase 3) — token counting, accumulation, ledger, spend cap."""

from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from factory_floor.cost import (
    CostTrackingCallback,
    DailyLedger,
    SpendCapExceeded,
    UsageAccumulator,
    check_spend_cap,
    count_tokens,
    estimate_cost,
)


class TestTokenCountingAndPricing:
    def test_count_tokens(self):
        assert count_tokens("") == 0
        assert count_tokens("hello world, this is a test") > 3

    def test_estimate_cost_uses_the_pricing_table(self):
        assert estimate_cost(1_000_000, 0, "gpt-4.1-mini") == pytest.approx(0.40)
        assert estimate_cost(0, 1_000_000, "gpt-4.1-mini") == pytest.approx(1.60)
        assert estimate_cost(0, 1_000_000, "text-embedding-3-small") == pytest.approx(0.0)

    def test_unknown_model_is_not_free(self):
        assert estimate_cost(1_000_000, 0, "some-future-model") > 0


class TestUsageAccumulator:
    def test_add_tracks_totals_and_per_model(self):
        acc = UsageAccumulator()
        acc.add(100, 20, "gpt-4.1-mini")
        acc.add(50, 10, "gpt-4.1-mini")
        assert acc.total_input_tokens == 150
        assert acc.total_output_tokens == 30
        assert acc.n_calls == 2
        assert acc.by_model["gpt-4.1-mini"]["n_calls"] == 2
        assert acc.total_usd == pytest.approx(estimate_cost(150, 30, "gpt-4.1-mini"))

    def test_dict_roundtrip(self):
        acc = UsageAccumulator()
        acc.add(100, 20, "gpt-4.1-mini")
        restored = UsageAccumulator.from_dict(acc.as_dict())
        assert restored.as_dict() == acc.as_dict()

    def test_merge_accumulates_across_turns(self):
        a = UsageAccumulator()
        a.add(100, 20, "gpt-4.1-mini")
        b = UsageAccumulator()
        b.add(30, 5, "gpt-4.1-mini")
        a.merge(b.as_dict())
        assert a.n_calls == 2
        assert a.total_input_tokens == 130
        assert a.by_model["gpt-4.1-mini"]["output_tokens"] == 25

    def test_from_dict_none_is_empty(self):
        acc = UsageAccumulator.from_dict(None)
        assert acc.n_calls == 0 and acc.total_usd == 0.0


class TestCheckSpendCap:
    def test_no_cap_is_a_noop(self):
        check_spend_cap(999.0, cap=None)

    def test_below_cap_does_not_raise(self):
        check_spend_cap(1.0, cap=10.0, pending_estimate=0.05, on_alert=None)

    def test_at_or_over_cap_raises(self):
        with pytest.raises(SpendCapExceeded):
            check_spend_cap(9.99, cap=10.0, pending_estimate=0.05, on_alert=None)

    def test_alert_fires_at_threshold(self):
        seen = []
        check_spend_cap(8.5, cap=10.0, pending_estimate=0.0, alert_threshold=0.8,
                        on_alert=lambda spent, cap: seen.append((spent, cap)))
        assert seen == [(8.5, 10.0)]

    def test_alert_does_not_fire_below_threshold(self):
        seen = []
        check_spend_cap(1.0, cap=10.0, alert_threshold=0.8, on_alert=lambda *a: seen.append(a))
        assert seen == []


class TestCostTrackingCallback:
    def test_reads_usage_metadata(self):
        acc = UsageAccumulator()
        cb = CostTrackingCallback(acc, default_model="gpt-4.1-mini")
        gen = SimpleNamespace(
            text="",
            message=SimpleNamespace(
                usage_metadata={"input_tokens": 100, "output_tokens": 20},
                response_metadata={"model_name": "gpt-4.1-mini"},
            ),
        )
        cb.on_llm_end(SimpleNamespace(generations=[[gen]], llm_output=None))
        assert acc.total_input_tokens == 100
        assert acc.total_output_tokens == 20
        assert acc.n_calls == 1

    def test_falls_back_to_counting_the_completion_text(self):
        acc = UsageAccumulator()
        cb = CostTrackingCallback(acc, default_model="gpt-4.1-mini")
        gen = SimpleNamespace(text="a fairly long completion string that should tokenize to several tokens",
                              message=SimpleNamespace(usage_metadata=None, response_metadata={}))
        cb.on_llm_end(SimpleNamespace(generations=[[gen]], llm_output=None))
        assert acc.n_calls == 1
        assert acc.total_input_tokens == 0
        assert acc.total_output_tokens > 0


class TestDailyLedger:
    def test_record_then_today_total_filters_by_tenant_and_date(self, tmp_path):
        ledger = DailyLedger(tmp_path / "ledger.jsonl")
        with freeze_time("2026-08-27 10:00:00"):
            acc = UsageAccumulator()
            acc.add(1_000_000, 0, "gpt-4.1-mini")  # $0.40
            ledger.record(tenant_id="default", usage=acc)
            ledger.record(tenant_id="other", usage=acc)
        with freeze_time("2026-08-28 10:00:00"):
            assert ledger.today_total("default") == 0.0  # yesterday's rows don't count
        with freeze_time("2026-08-27 23:00:00"):
            assert ledger.today_total("default") == pytest.approx(0.40)
            assert ledger.today_total("other") == pytest.approx(0.40)

    def test_missing_file_is_zero(self, tmp_path):
        assert DailyLedger(tmp_path / "nope.jsonl").today_total() == 0.0
