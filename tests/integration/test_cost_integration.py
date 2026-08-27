"""Phase 3: cost metering + spend cap through services.run_diagnostic, end to end
against the real create_agent graph with a fake model — no API key, no cost.
"""

import json

import pytest
from freezegun import freeze_time

from factory_floor import services

pytestmark = pytest.mark.integration


def _request():
    return services.DiagnosticRequest(
        question_text="what does F30021 mean?",
        machine_id="GENERAL",
        equipment_type="VFD",
        language="English",
    )


def test_a_turn_writes_a_ledger_row_and_populates_result_cost(
    tmp_path, monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    ledger_path = tmp_path / "cost_ledger.jsonl"
    monkeypatch.setenv("FACTORY_FLOOR_COST_LEDGER_PATH", str(ledger_path))
    # no cap -> nothing is blocked, but cost is still tracked

    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("Ground fault.")
    )

    assert result.blocked is False
    assert result.cost is not None
    assert result.cost["n_calls"] >= 1  # the agent's model call registered (fallback counting)
    assert ledger_path.exists()
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["tenant_id"] == "default"
    assert rows[0]["n_calls"] >= 1


def test_spend_cap_blocks_before_the_agent_runs(tmp_path, monkeypatch, tmp_vectorstore, make_agent_fake_llm):
    ledger_path = tmp_path / "cost_ledger.jsonl"
    # Pre-seed today's ledger with spend already over the cap.
    with freeze_time("2026-08-27 09:00:00"):
        ledger_path.write_text(
            json.dumps({"date": "2026-08-27", "tenant_id": "default", "usd": 9.99}) + "\n"
        )
        monkeypatch.setenv("FACTORY_FLOOR_COST_LEDGER_PATH", str(ledger_path))
        monkeypatch.setenv("FACTORY_FLOOR_DAILY_SPEND_CAP_USD", "10.00")

        result = services.run_diagnostic(
            _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("should not be reached")
        )

    assert result.blocked is True
    assert result.answer is None
    assert "cap" in result.message.lower()


def test_spend_cap_block_in_streaming_mode_returns_empty_generator(
    tmp_path, monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    ledger_path = tmp_path / "cost_ledger.jsonl"
    with freeze_time("2026-08-27 09:00:00"):
        ledger_path.write_text(
            json.dumps({"date": "2026-08-27", "tenant_id": "default", "usd": 50.0}) + "\n"
        )
        monkeypatch.setenv("FACTORY_FLOOR_COST_LEDGER_PATH", str(ledger_path))
        monkeypatch.setenv("FACTORY_FLOOR_DAILY_SPEND_CAP_USD", "10.00")

        generator, result = services.run_diagnostic(
            _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("x"), stream=True
        )

    assert result.blocked is True
    assert list(generator) == []


def test_streaming_turn_finalizes_cost_only_after_consumption(
    tmp_path, monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    monkeypatch.setenv("FACTORY_FLOOR_COST_LEDGER_PATH", str(tmp_path / "cost_ledger.jsonl"))
    generator, result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("Ground fault."), stream=True
    )
    assert result.cost is None
    list(generator)
    assert result.cost is not None and result.cost["n_calls"] >= 1
