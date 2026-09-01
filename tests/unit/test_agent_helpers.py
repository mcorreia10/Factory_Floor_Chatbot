"""Pure-logic tests for factory_floor/agent.py helpers — the message/trace plumbing,
not the agent loop itself (that needs a real model; see the llm-marked wiring test).
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from factory_floor.agent import (
    _extract_tool_trace,
    format_history,
    format_machine_context,
    source_list_from_docs,
)


class TestFormatMachineContext:
    def test_general_and_blank_return_empty(self):
        assert format_machine_context("GENERAL") == ""
        assert format_machine_context("") == ""

    def test_unknown_id_returns_empty(self):
        assert format_machine_context("NOPE-999") == ""

    def test_real_machine_is_described_without_asking_what_it_is(self):
        # VFD-01 is in the committed machines.csv.
        out = format_machine_context("VFD-01")
        assert "VFD-01" in out
        assert "variable-frequency drive" in out
        assert "SINAMICS G120C" in out


class TestFormatHistory:
    def test_empty_history(self):
        assert format_history([]) == "No recorded maintenance history for this machine."

    def test_uses_fault_code_when_present_else_description(self):
        rows = [
            {"event_date": "2022-06-25", "event_type": "corrective", "fault_code": "F30021",
             "description": "ground fault", "action_taken": "replaced motor cable"},
            {"event_date": "2021-01-02", "event_type": "preventive_maintenance", "fault_code": "",
             "description": "bearing noise", "action_taken": "re-greased"},
        ]
        out = format_history(rows)
        assert "2022-06-25 [corrective] F30021 — action taken: replaced motor cable" in out
        assert "2021-01-02 [preventive_maintenance] bearing noise — action taken: re-greased" in out

    def test_caps_at_max_rows_keeping_the_most_recent(self):
        # operator_resolution events accumulate without bound (difficulty #18 fix), so
        # the history must not grow the agent's prompt forever.
        rows = [
            {"event_date": f"2020-01-{day:02d}", "event_type": "fault", "fault_code": f"F{day:05d}",
             "description": "", "action_taken": "x"}
            for day in range(1, 26)
        ]
        out = format_history(rows, max_rows=10)
        assert "15 older event(s) omitted" in out
        assert "F00025" in out       # newest kept
        assert "F00016" in out       # 10th newest kept
        assert "F00015" not in out   # 11th newest dropped
        assert "F00001" not in out   # oldest dropped

    def test_sorts_oldest_first_regardless_of_input_order(self):
        rows = [
            {"event_date": "2024-05-05", "event_type": "fault", "fault_code": "LATER",
             "description": "", "action_taken": "x"},
            {"event_date": "2021-01-01", "event_type": "fault", "fault_code": "EARLIER",
             "description": "", "action_taken": "x"},
        ]
        out = format_history(rows)
        assert out.index("EARLIER") < out.index("LATER")


class TestSourceListFromDocs:
    def test_dedupes_by_file_and_page_and_skips_not_found(self):
        docs = [
            Document(page_content="a", metadata={"source_file": "M.pdf", "page": 907}),
            Document(page_content="b", metadata={"source_file": "M.pdf", "page": 907}),  # dup
            Document(page_content="c", metadata={"source_file": "M.pdf", "page": 61}),
            Document(page_content="notice", metadata={"source_file": "(not in corpus)", "page": -1, "not_found": True}),
        ]
        out = source_list_from_docs(docs)
        assert out == "- M.pdf — page 908\n- M.pdf — page 62"


class TestExtractToolTrace:
    def test_pairs_each_tool_call_with_its_result(self):
        messages = [
            HumanMessage(content="why F30021?"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "search_manuals", "args": {"query": "F30021 ground fault"}}],
            ),
            ToolMessage(content="[SOURCE 1] ListManual.pdf, page 908 ...", tool_call_id="call_1", name="search_manuals"),
            AIMessage(content="It is a ground fault."),
        ]
        trace = _extract_tool_trace(messages)
        assert len(trace) == 1
        assert trace[0]["tool"] == "search_manuals"
        assert trace[0]["input"] == {"query": "F30021 ground fault"}
        assert trace[0]["output_preview"].startswith("[SOURCE 1] ListManual.pdf")

    def test_no_tool_calls_yields_empty_trace(self):
        messages = [HumanMessage(content="hi"), AIMessage(content="general answer, no tools")]
        assert _extract_tool_trace(messages) == []
