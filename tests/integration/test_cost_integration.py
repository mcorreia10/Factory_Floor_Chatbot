"""Phase 3 (+ phase 5 storage): cost metering + spend cap through
services.run_diagnostic, end to end against the real create_agent graph with a fake
model. The cost ledger is the SQLite ``cost_events`` table now; the audit db path is
pointed at tmp_path by the conftest isolation fixture.
"""

from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from factory_floor import audit, services

pytestmark = pytest.mark.integration


def _request():
    return services.DiagnosticRequest(
        question_text="what does F30021 mean?",
        machine_id="GENERAL",
        equipment_type="VFD",
        language="English",
    )


def _seed_spend(usd):
    audit.DailyLedger().record(
        tenant_id="default",
        usage=SimpleNamespace(total_usd=usd, total_input_tokens=0, total_output_tokens=0, n_calls=1),
    )


def test_a_turn_records_cost_and_an_audit_row(tmp_vectorstore, make_agent_fake_llm):
    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("Ground fault.")
    )

    assert result.blocked is False
    assert result.cost is not None and result.cost["n_calls"] >= 1
    assert result.audit_id is not None  # audit_enabled is the default

    trail = audit.get_audit_trail(machine_id="GENERAL")
    assert len(trail) == 1
    assert trail[0]["id"] == result.audit_id
    assert trail[0]["safety_action"] == "pass"
    # the linked cost_events row feeds the daily total
    assert audit.DailyLedger().today_total("default") > 0


def test_spend_cap_blocks_before_the_agent_runs(monkeypatch, tmp_vectorstore, make_agent_fake_llm):
    # setenv before anything reads settings, so get_settings() caches with the cap set.
    monkeypatch.setenv("FACTORY_FLOOR_DAILY_SPEND_CAP_USD", "10.00")
    with freeze_time("2026-08-27 09:00:00"):
        _seed_spend(9.99)
        result = services.run_diagnostic(
            _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("should not be reached")
        )

    assert result.blocked is True
    assert result.answer is None
    assert "cap" in result.message.lower()


def test_spend_cap_block_in_streaming_mode_returns_empty_generator(
    monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    monkeypatch.setenv("FACTORY_FLOOR_DAILY_SPEND_CAP_USD", "10.00")
    with freeze_time("2026-08-27 09:00:00"):
        _seed_spend(50.0)
        generator, result = services.run_diagnostic(
            _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("x"), stream=True
        )

    assert result.blocked is True
    assert list(generator) == []


def test_streaming_turn_finalizes_cost_only_after_consumption(tmp_vectorstore, make_agent_fake_llm):
    generator, result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("Ground fault."), stream=True
    )
    assert result.cost is None
    list(generator)
    assert result.cost is not None and result.cost["n_calls"] >= 1
    assert result.audit_id is not None
