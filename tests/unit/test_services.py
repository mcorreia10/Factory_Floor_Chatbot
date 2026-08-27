"""Service layer (phase 2). The agent loop and the classifier are stubbed — what's under
test is the wiring: request -> retriever -> agent -> result -> turn dict.
"""

from unittest.mock import MagicMock

from factory_floor import services


class TestCheckTypo:
    def test_delegates_to_fault_codes(self):
        assert services.check_typo("display reads F3OO21") == [("F3OO21", "F30021")]

    def test_none_and_empty(self):
        assert services.check_typo(None) == []
        assert services.check_typo("") == []


class TestClassifyPhoto:
    def test_no_classifier_returns_none(self):
        assert services.classify_photo(b"anything", clf=None) is None

    def test_builds_the_vision_context_summary(self, monkeypatch):
        canned = {"predicted_label": "structural_damage", "confidence": 0.91, "is_defective": True}
        monkeypatch.setattr(services, "classify_defect_trained", lambda *a, **k: canned)
        out = services.classify_photo(b"fakebytes", clf=object())
        assert out["classification"] is canned
        assert "structural_damage" in out["vision_context"]
        assert "91%" in out["vision_context"]
        assert "defective" in out["vision_context"]

    def test_no_defect_phrasing(self, monkeypatch):
        canned = {"predicted_label": "good", "confidence": 0.8, "is_defective": False}
        monkeypatch.setattr(services, "classify_defect_trained", lambda *a, **k: canned)
        out = services.classify_photo(b"x", clf=object())
        assert "no defect detected" in out["vision_context"]


class TestBuildDiagnosticRetriever:
    def test_returns_something_invokable_without_hitting_the_network(self, fake_llm):
        retriever = services.build_diagnostic_retriever(MagicMock(), "VFD", fake_llm, k=5)
        assert hasattr(retriever, "invoke")


class TestRunDiagnostic:
    def test_blocking_maps_agent_dict_onto_a_result(self, monkeypatch, fake_llm):
        canned = {
            "question": "why F30021?",
            "answer": "Earth fault suspected.",
            "documents": [],
            "sources": "- ListManual.pdf — page 908",
            "tool_trace": [{"tool": "search_manuals", "input": {"query": "F30021"}}],
            "language": "English",
            "run_id": "run-123",
        }
        monkeypatch.setattr(services, "run_diagnostic_agent", lambda *a, **k: canned)
        req = services.DiagnosticRequest(question_text="why F30021?", machine_id="GENERAL")
        result = services.run_diagnostic(req, vectorstore=MagicMock(), llm=fake_llm)
        assert isinstance(result, services.DiagnosticResult)
        assert result.answer == "Earth fault suspected."
        assert result.run_id == "run-123"
        assert result.cache_hit is False and result.blocked is False
        # safety gate ran (phase 4) — a non-action answer passes on the cheap keyword path.
        assert result.safety["action"] == "pass"
        # cost is always a dict now (phase 3); the stubbed agent never invokes the callback.
        assert result.cost["n_calls"] == 0

    def test_streaming_fills_the_result_only_after_the_generator_is_consumed(self, monkeypatch, fake_llm):
        # gate off -> raw tokens pass straight through (the gated path is covered in
        # test_safety_gate_integration.py).
        monkeypatch.setenv("FACTORY_FLOOR_SAFETY_GATE_MODE", "off")

        class FakeStreamed:
            run_id = "run-xyz"
            answer = None
            documents: list = []
            sources = None
            tool_trace = None

        fs = FakeStreamed()

        def fake_stream(*args, **kwargs):
            def gen():
                yield "Ground "
                yield "fault."
                fs.answer = "Ground fault."
                fs.documents = ["doc"]
                fs.sources = "- s"
                fs.tool_trace = []

            return gen(), fs

        monkeypatch.setattr(services, "stream_diagnostic_agent", fake_stream)
        req = services.DiagnosticRequest(question_text="q", machine_id="GENERAL")
        generator, result = services.run_diagnostic(req, vectorstore=MagicMock(), llm=fake_llm, stream=True)

        assert result.answer is None  # nothing consumed yet
        assert list(generator) == ["Ground ", "fault."]
        assert result.answer == "Ground fault."
        assert result.documents == ["doc"]
        assert result.run_id == "run-xyz"


class TestAssembleTurn:
    def _result(self):
        return services.DiagnosticResult(
            question="why F30021?",
            answer="Earth fault suspected.",
            documents=["d1"],
            sources="- ListManual.pdf — page 908",
            tool_trace=[{"tool": "search_manuals", "input": {}}],
            run_id="r1",
            language="English",
            safety={"action": "pass", "reason": "no physical action instructed"},
        )

    def test_text_turn_shape(self):
        turn = services.assemble_turn(
            self._result(), image_bytes=None, classification=None, vision_context=None, language="English"
        )
        assert turn == {
            "type": "agent",
            "question": "why F30021?",
            "answer": "Earth fault suspected.",
            "documents": ["d1"],
            "sources": "- ListManual.pdf — page 908",
            "tool_trace": [{"tool": "search_manuals", "input": {}}],
            "safety": {"action": "pass", "reason": "no physical action instructed"},
            "image_bytes": None,
            "vision_context": None,
            "predicted_label": None,
            "is_defective": None,
            "language": "English",
        }

    def test_photo_turn_pulls_label_and_defect_flag_from_classification(self):
        result = self._result()
        result.question = ""
        turn = services.assemble_turn(
            result,
            image_bytes=b"img",
            classification={"predicted_label": "contamination", "is_defective": True},
            vision_context="Vision analysis: contamination",
            language="Portuguese",
        )
        assert turn["question"] == "[Uploaded a photo of a component]"
        assert turn["image_bytes"] == b"img"
        assert turn["predicted_label"] == "contamination"
        assert turn["is_defective"] is True
        assert turn["language"] == "Portuguese"
