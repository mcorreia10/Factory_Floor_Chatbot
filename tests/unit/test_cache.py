"""factory_floor/cache.py (phase 6) — the semantic answer cache.

Built with DeterministicFakeEmbedding so no network; the Chroma dir is tmp_path (via the
conftest isolation fixture).
"""

from types import SimpleNamespace

import pytest
from freezegun import freeze_time
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from factory_floor.cache import SemanticCache, _code_signature, _normalize_question


@pytest.fixture
def cache():
    return SemanticCache(embeddings=DeterministicFakeEmbedding(size=64))


def _result(answer="Isolate, then check the cable.", action="pass", **over):
    base = dict(
        answer=answer,
        sources="- LM.pdf — page 908",
        documents=[SimpleNamespace(page_content="F30021 def", metadata={"source_file": "LM.pdf", "page": 907})],
        tool_trace=[{"tool": "search_manuals", "input": {"query": "F30021"}}],
        run_id="run-1",
        safety={"action": action},
        blocked=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestHelpers:
    def test_normalize_question_collapses_whitespace_and_case(self):
        assert _normalize_question("  The  MOTOR  is\nHOT ") == "the motor is hot"

    def test_code_signature_sorted_and_normalized(self):
        assert _code_signature("saw F30022 then F30021") == "F30021,F30022"
        assert _code_signature("F3OO21 on the display") == "F30021"
        assert _code_signature("the motor is overheating") == ""


class TestStoreRefusals:
    def test_refuses_held_answers(self, cache):
        assert cache.store("q", _result(action="held"), machine_id="GENERAL", equipment_type="VFD") is False

    def test_refuses_blocked_and_empty(self, cache):
        assert cache.store("q", _result(blocked=True)) is False
        assert cache.store("q", _result(answer="   ")) is False

    def test_stores_pass_and_rewritten(self, cache):
        assert cache.store("q1", _result(action="pass"), equipment_type="VFD") is True
        assert cache.store("q2", _result(action="rewritten"), equipment_type="VFD") is True
        assert cache.count() == 2


class TestLookup:
    def test_prose_exact_hit_rehydrates_documents(self, cache):
        cache.store("the motor is overheating and vibrating", _result(), machine_id="MOTOR-01",
                    equipment_type="electric_motor")
        hit = cache.lookup("The motor is  overheating and vibrating", machine_id="MOTOR-01",
                           equipment_type="electric_motor")
        assert hit is not None
        assert hit.similarity == pytest.approx(1.0, abs=1e-6)
        assert hit.answer.startswith("Isolate")
        assert hit.documents[0].metadata["source_file"] == "LM.pdf"
        assert hit.tool_trace[0]["tool"] == "search_manuals"

    def test_scope_isolation(self, cache):
        cache.store("bearing noise", _result(), machine_id="MOTOR-01", equipment_type="electric_motor")
        assert cache.lookup("bearing noise", machine_id="MOTOR-02", equipment_type="electric_motor") is None
        assert cache.lookup("bearing noise", machine_id="MOTOR-01", equipment_type="VFD") is None
        assert cache.lookup("bearing noise", machine_id="MOTOR-01", equipment_type="electric_motor", language="French") is None

    def test_fault_code_question_needs_exact_match_not_a_near_miss(self, cache):
        cache.store("F30021 ground fault after several hours running", _result(),
                    machine_id="GENERAL", equipment_type="VFD")
        assert cache.lookup("F30021 ground fault after several hours running",
                            machine_id="GENERAL", equipment_type="VFD") is not None
        # same sentence, different code -> must NOT be served
        assert cache.lookup("F30022 ground fault after several hours running",
                            machine_id="GENERAL", equipment_type="VFD") is None
        # code-free paraphrase of a code-bearing entry -> also not served
        assert cache.lookup("ground fault after several hours running",
                            machine_id="GENERAL", equipment_type="VFD") is None

    def test_ttl_expiry(self, cache):
        with freeze_time("2026-08-27 10:00:00"):
            cache.store("compressor won't start", _result(), equipment_type="VFD")
        with freeze_time("2026-08-27 20:00:00"):
            assert cache.lookup("compressor won't start", equipment_type="VFD") is not None
        with freeze_time("2026-10-01 10:00:00"):  # well past the 720h default TTL
            assert cache.lookup("compressor won't start", equipment_type="VFD") is None

    def test_version_stamp_mismatch_is_ignored(self, cache, monkeypatch):
        cache.store("overvoltage trip", _result(), equipment_type="VFD")
        assert cache.lookup("overvoltage trip", equipment_type="VFD") is not None
        monkeypatch.setenv("FACTORY_FLOOR_EMBEDDING_MODEL", "text-embedding-3-large")
        from factory_floor.config import get_settings

        get_settings.cache_clear()
        assert cache.lookup("overvoltage trip", equipment_type="VFD") is None


class TestClear:
    def test_clear_empties_the_collection(self, cache):
        cache.store("q", _result(), equipment_type="VFD")
        assert cache.count() == 1
        cache.clear()
        assert cache.count() == 0
