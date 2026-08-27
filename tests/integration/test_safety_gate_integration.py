"""Phase 4: how the safety gate plugs into services.run_diagnostic.

enforce_safety's own logic is unit-tested (test_safety_gate.py); here it is stubbed so
the test controls the verdict and checks the wiring: the delivered answer is the gate's,
result.safety carries the verdict, and mode=off keeps live token streaming.
"""

import pytest

from factory_floor import services
from factory_floor.safety import SafetyGateResult

pytestmark = pytest.mark.integration


def _request():
    return services.DiagnosticRequest(
        question_text="how do I check the ground fault?",
        machine_id="GENERAL",
        equipment_type="VFD",
        language="English",
    )


def test_rewritten_answer_replaces_result_answer_and_is_recorded(
    monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    gate = SafetyGateResult(
        action="rewritten",
        original_answer="check the cable",
        delivered_answer="Safety precautions: de-energize first. Then check the cable.",
        audit={"judge": {"passed": False}, "recheck": {"passed": True}},
        reason="rewrite adds a precautions-first section",
    )
    monkeypatch.setattr(services, "enforce_safety", lambda *a, **k: gate)

    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("check the cable")
    )
    assert result.answer == gate.delivered_answer
    assert result.safety["action"] == "rewritten"


def test_held_answer_delivers_the_fallback(monkeypatch, tmp_vectorstore, make_agent_fake_llm):
    gate = SafetyGateResult("held", "unsafe original", "SAFE FALLBACK TEXT", {}, "held")
    monkeypatch.setattr(services, "enforce_safety", lambda *a, **k: gate)

    result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("unsafe original")
    )
    assert result.answer == "SAFE FALLBACK TEXT"
    assert result.safety["action"] == "held"


def test_streaming_with_gate_active_emits_the_gated_text_once(
    monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    gate = SafetyGateResult("rewritten", "raw", "GATED ANSWER", {}, "r")
    monkeypatch.setattr(services, "enforce_safety", lambda *a, **k: gate)

    generator, result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("raw tokens here"), stream=True
    )
    assert result.answer is None
    chunks = list(generator)
    assert chunks == ["GATED ANSWER"]  # drained silently, delivered once
    assert result.answer == "GATED ANSWER"
    assert result.safety["action"] == "rewritten"


def test_streaming_with_gate_off_streams_raw_tokens(
    monkeypatch, tmp_vectorstore, make_agent_fake_llm
):
    monkeypatch.setenv("FACTORY_FLOOR_SAFETY_GATE_MODE", "off")

    generator, result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=make_agent_fake_llm("Ground fault on the power unit."), stream=True
    )
    streamed = "".join(generator)
    # With the gate off the inner agent stream passes straight through (>=1 chunk),
    # and the final answer is unchanged.
    assert "Ground fault" in streamed
    assert result.answer == "Ground fault on the power unit."
    assert result.safety["action"] == "pass"
