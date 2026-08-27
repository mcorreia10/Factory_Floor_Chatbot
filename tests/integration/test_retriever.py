"""Integration tests for the retriever stack against a real (tiny) Chroma store built
with deterministic fake embeddings — no API key, no network.

Focus: the lexical fault-code path (`lookup_code`, `CodeAwareRetriever`), which is
embedding-independent and is the part most worth locking down.
"""

import pytest

from factory_floor.fault_codes import lookup_code
from factory_floor.rag import CodeAwareRetriever, build_retriever

pytestmark = pytest.mark.integration


class TestLookupCode:
    def test_literal_hit_ranks_the_definition_above_the_cross_reference(self, tmp_vectorstore):
        docs = lookup_code(tmp_vectorstore, "F30021")
        assert len(docs) == 2  # the definition page and the "See also" page
        assert docs[0].metadata["page"] == 907  # definition first, per score_occurrence

    def test_equipment_type_filter_excludes_other_families(self, tmp_vectorstore):
        assert lookup_code(tmp_vectorstore, "F30021", equipment_type="electric_motor") == []
        assert lookup_code(tmp_vectorstore, "F30021", equipment_type="VFD")

    def test_unknown_code_has_no_literal_hits(self, tmp_vectorstore):
        assert lookup_code(tmp_vectorstore, "F99999") == []


class TestCodeAwareRetriever:
    def test_bare_known_code_returns_pinned_literal_hits_only(self, tmp_vectorstore):
        retriever = build_retriever(tmp_vectorstore, k=5, code_aware=True)
        assert isinstance(retriever, CodeAwareRetriever)
        docs = retriever.invoke("F30021")
        assert docs
        assert all("F30021" in d.page_content for d in docs)

    def test_bare_unknown_code_returns_a_not_found_notice_not_a_wrong_page(self, tmp_vectorstore):
        retriever = build_retriever(tmp_vectorstore, k=5, code_aware=True)
        docs = retriever.invoke("F99999")
        assert len(docs) == 1
        assert docs[0].metadata.get("not_found") is True
        assert "F99999" in docs[0].page_content

    def test_no_code_delegates_to_the_semantic_path(self, tmp_vectorstore):
        retriever = build_retriever(tmp_vectorstore, k=3, code_aware=True)
        docs = retriever.invoke("bearing noise and insulation resistance")
        assert docs  # deterministic fake embeddings still return *something*
        assert all(not d.metadata.get("not_found") for d in docs)
