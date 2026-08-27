"""Phase 6: the semantic cache short-circuits services.run_diagnostic on a repeat
first-turn question — no agent call, no cost — while still writing an audit row.
"""

import pytest

from factory_floor import audit, services

pytestmark = pytest.mark.integration


def _req(question="the motor is overheating and vibrating, what should I check?",
         machine_id="MOTOR-01", equipment_type="electric_motor"):
    return services.DiagnosticRequest(
        question_text=question, machine_id=machine_id, equipment_type=equipment_type,
        operator_id="OP-1001", language="English",
    )


@pytest.fixture(autouse=True)
def _enable_cache(monkeypatch):
    monkeypatch.setenv("FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED", "true")
    # gate off: the agent fake can't do structured output; the gate's own behaviour is
    # covered in test_safety_gate*.py.
    monkeypatch.setenv("FACTORY_FLOOR_SAFETY_GATE_MODE", "off")


def test_repeat_question_is_served_from_cache_with_no_agent_call(tmp_vectorstore, make_agent_fake_llm):
    llm = make_agent_fake_llm("Check ventilation and bearing condition.")

    first = services.run_diagnostic(_req(), vectorstore=tmp_vectorstore, llm=llm)
    assert first.cache_hit is False
    assert first.cost["n_calls"] >= 1

    second = services.run_diagnostic(_req(), vectorstore=tmp_vectorstore, llm=llm)
    assert second.cache_hit is True
    assert second.cost["n_calls"] == 0  # the agent (and its cost callback) never ran
    assert second.answer == first.answer
    assert second.run_id == first.run_id  # carried from the cached entry

    # a cache hit still gets an audit row, flagged
    trail = audit.get_audit_trail(machine_id="MOTOR-01")
    assert len(trail) == 2
    assert trail[0]["cache_hit"] == 1


def test_other_machine_is_a_miss(tmp_vectorstore, make_agent_fake_llm):
    llm = make_agent_fake_llm("Answer.")
    services.run_diagnostic(_req(machine_id="MOTOR-01"), vectorstore=tmp_vectorstore, llm=llm)
    other = services.run_diagnostic(_req(machine_id="MOTOR-02"), vectorstore=tmp_vectorstore, llm=llm)
    assert other.cache_hit is False


def test_fault_code_question_only_hits_on_the_exact_code(tmp_vectorstore, make_agent_fake_llm):
    llm = make_agent_fake_llm("F30021 is a ground fault.")
    q1 = _req("F30021 ground fault after several hours running", machine_id="GENERAL", equipment_type="VFD")
    services.run_diagnostic(q1, vectorstore=tmp_vectorstore, llm=llm)

    same = services.run_diagnostic(q1, vectorstore=tmp_vectorstore, llm=llm)
    assert same.cache_hit is True

    q2 = _req("F30022 ground fault after several hours running", machine_id="GENERAL", equipment_type="VFD")
    other_code = services.run_diagnostic(q2, vectorstore=tmp_vectorstore, llm=llm)
    assert other_code.cache_hit is False


def test_streaming_cache_hit_yields_the_answer_once(tmp_vectorstore, make_agent_fake_llm):
    llm = make_agent_fake_llm("Bearing wear likely; inspect and re-grease.")
    services.run_diagnostic(_req(), vectorstore=tmp_vectorstore, llm=llm)

    generator, result = services.run_diagnostic(_req(), vectorstore=tmp_vectorstore, llm=llm, stream=True)
    assert result.cache_hit is True
    assert list(generator) == [result.answer]


def test_disabled_cache_never_hits(monkeypatch, tmp_vectorstore, make_agent_fake_llm):
    monkeypatch.setenv("FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED", "false")
    llm = make_agent_fake_llm("Answer.")
    services.run_diagnostic(_req(), vectorstore=tmp_vectorstore, llm=llm)
    again = services.run_diagnostic(_req(), vectorstore=tmp_vectorstore, llm=llm)
    assert again.cache_hit is False
