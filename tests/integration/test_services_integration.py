"""Phase 2: run a full diagnostic turn through the *real* create_agent graph and the
real retriever, against a tiny local Chroma store, with a fake model — no API key.

Proves the extraction is wired correctly: DiagnosticRequest -> build_diagnostic_retriever
-> stream/run_diagnostic_agent -> DiagnosticResult / streamed result / turn dict.
"""

import pytest

from factory_floor import services

pytestmark = pytest.mark.integration


def _request():
    return services.DiagnosticRequest(
        question_text="what does F30021 mean?",
        machine_id="GENERAL",
        equipment_type="VFD",
        chat_history=[],
        language="English",
    )


def test_blocking_turn_end_to_end(tmp_vectorstore, make_agent_fake_llm):
    # Gate-neutral answer (no instructed physical action) so the default safety gate
    # passes it on the cheap keyword path without an LLM judge call.
    llm = make_agent_fake_llm("F30021 indicates a power-unit ground fault.")
    result = services.run_diagnostic(_request(), vectorstore=tmp_vectorstore, llm=llm)

    assert isinstance(result, services.DiagnosticResult)
    assert result.answer == "F30021 indicates a power-unit ground fault."
    assert result.safety["action"] == "pass"
    assert result.tool_trace == []  # the fake model does not call tools
    assert result.run_id
    assert result.blocked is False and result.cache_hit is False


def test_streaming_turn_end_to_end_then_assemble_turn(tmp_vectorstore, make_agent_fake_llm):
    llm = make_agent_fake_llm("Ground fault on the power unit.")
    generator, result = services.run_diagnostic(
        _request(), vectorstore=tmp_vectorstore, llm=llm, stream=True
    )

    assert result.answer is None  # not consumed yet
    streamed_text = "".join(generator)
    assert "Ground fault" in streamed_text
    assert result.answer == "Ground fault on the power unit."

    turn = services.assemble_turn(
        result, image_bytes=None, classification=None, vision_context=None, language="English"
    )
    assert turn["type"] == "agent"
    assert turn["question"] == "what does F30021 mean?"
    assert turn["answer"] == "Ground fault on the power unit."


def test_general_question_does_not_offer_the_history_tool(tmp_vectorstore, make_agent_fake_llm):
    # A GENERAL machine_id means get_maintenance_history must not even be available.
    from factory_floor.agent import build_diagnostic_agent
    from factory_floor.rag import build_retriever

    retriever = build_retriever(tmp_vectorstore, k=3, code_aware=True)
    _agent, _docs, tools = build_diagnostic_agent(retriever, "GENERAL", llm=make_agent_fake_llm("x"))
    assert [t.name for t in tools] == ["search_manuals"]
