"""Integration test for factory_floor/agent._build_agent_run — the setup shared by the
blocking and streaming entry points: message assembly, tool availability, and the
LangSmith run config.

The agent *loop* (create_agent driving a real tool-calling model) is not exercised here
— that needs a real model and is covered by an llm-marked test.
"""

import pytest

from factory_floor.agent import _build_agent_run

pytestmark = pytest.mark.integration


def _run(tmp_vectorstore, machine_id, *, vision_context=None, language="English", make_fake_llm):
    from factory_floor.rag import build_retriever

    retriever = build_retriever(tmp_vectorstore, k=3, code_aware=True)
    return _build_agent_run(
        question_text="the drive tripped on F30021",
        retriever=retriever,
        machine_id=machine_id,
        chat_history=[],
        vision_context=vision_context,
        llm=make_fake_llm("unused – the loop is not run here"),
        language=language,
        config=None,
    )


class TestBuildAgentRun:
    def test_system_prompt_first_human_last_and_language_substituted(self, tmp_vectorstore, make_fake_llm):
        _agent, _docs, messages, _cfg = _run(tmp_vectorstore, "VFD-01", language="Portuguese", make_fake_llm=make_fake_llm)
        assert messages[0].type == "system"
        assert "Respond in Portuguese" in messages[0].content
        assert messages[-1].type == "human"
        assert "F30021" in messages[-1].content

    def test_history_tool_offered_only_when_a_specific_machine_is_selected(self, tmp_vectorstore, make_fake_llm):
        _a, _d, _m, cfg_specific = _run(tmp_vectorstore, "VFD-01", make_fake_llm=make_fake_llm)
        assert cfg_specific["metadata"]["tools_available"] == ["search_manuals", "get_maintenance_history"]

        _a, _d, _m, cfg_general = _run(tmp_vectorstore, "GENERAL", make_fake_llm=make_fake_llm)
        assert cfg_general["metadata"]["tools_available"] == ["search_manuals"]

    def test_run_config_is_tagged_and_named_for_langsmith(self, tmp_vectorstore, make_fake_llm):
        _a, _d, _m, cfg = _run(tmp_vectorstore, "VFD-01", vision_context="Vision: good", make_fake_llm=make_fake_llm)
        assert cfg["run_name"] == "diagnostic_agent"
        assert "factory_floor" in cfg["tags"]
        assert "machine:VFD-01" in cfg["tags"]
        assert cfg["metadata"]["machine_id"] == "VFD-01"
        assert cfg["metadata"]["has_vision_context"] is True
        assert cfg["run_id"] is not None

    def test_machine_context_line_is_prepended_for_a_known_machine(self, tmp_vectorstore, make_fake_llm):
        _a, _d, messages, _c = _run(tmp_vectorstore, "VFD-01", make_fake_llm=make_fake_llm)
        assert "Machine selected by the operator: VFD-01" in messages[-1].content
