"""Phase 5: a diagnostic turn through services.run_diagnostic writes exactly one audit
record (with its sources, tool calls, cost, and safety verdict), and machines.py unions
live resolution events into the history.
"""

import pytest

from factory_floor import audit, services
from factory_floor.machines import append_resolution_event, get_machine_history

pytestmark = pytest.mark.integration


def _request():
    return services.DiagnosticRequest(
        question_text="what does F30021 mean?",
        machine_id="VFD-06",
        equipment_type="VFD",
        operator_id="OP-1001",
        tenant_id="default",
        language="English",
    )


def test_turn_writes_one_audit_recommendation_with_operator(tmp_vectorstore, make_agent_fake_llm):
    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("F30021 is a ground fault.")
    )
    assert result.audit_id is not None

    trail = audit.get_audit_trail(machine_id="VFD-06")
    assert len(trail) == 1
    row = trail[0]
    assert row["id"] == result.audit_id
    assert row["operator_id"] == "OP-1001"
    assert row["safety_action"] == "pass"
    assert row["run_id"] == result.run_id


def test_audit_disabled_writes_no_recommendation(monkeypatch, tmp_vectorstore, make_agent_fake_llm):
    monkeypatch.setenv("FACTORY_FLOOR_AUDIT_ENABLED", "false")
    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("F30021 is a ground fault.")
    )
    assert result.audit_id is None
    assert audit.get_audit_trail() == []


def test_resolution_event_shows_up_in_machine_history(tmp_vectorstore, make_agent_fake_llm):
    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("F30021 is a ground fault.")
    )
    append_resolution_event(
        "VFD-06", operator_id="OP-1001",
        steps_text="Isolated the drive, checked the motor cable, no fault found; reset.",
        recommendation_id=result.audit_id,
    )

    without = get_machine_history("VFD-06")
    with_live = get_machine_history("VFD-06", include_resolutions=True)
    assert len(with_live) == len(without) + 1
    live_row = [r for r in with_live if r["event_type"] == "operator_resolution"][0]
    assert live_row["technician"] == "OP-1001"
    assert "motor cable" in live_row["action_taken"]
    # the static CSV rows are unchanged
    assert all(r["event_type"] != "operator_resolution" for r in without)
