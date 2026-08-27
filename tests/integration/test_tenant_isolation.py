"""Phase 7: retrieval scoped to one tenant never returns another tenant's documents.

Two tiny Chroma collections in the same persist dir, built with fake embeddings.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from factory_floor.rag import build_retriever
from factory_floor.tenancy import resolve_collection
from factory_floor.vectorstore import build_vectorstore

pytestmark = pytest.mark.integration


def _store(tmp_path, tenant_id, marker):
    return build_vectorstore(
        [Document(page_content=f"{marker}: F30021 ground fault remedy for this tenant only",
                  metadata={"source_file": f"{marker}.pdf", "page": 1, "equipment_type": "VFD"})],
        tmp_path / "vs",  # one persist dir, different collection names
        resolve_collection(tenant_id),
        embeddings=DeterministicFakeEmbedding(size=64),
        rebuild=False,
    )


def test_each_tenant_only_sees_its_own_collection(tmp_path):
    acme = _store(tmp_path, "acme", "ACME")
    beta = _store(tmp_path, "beta", "BETA")

    assert acme._collection.name == "factory_floor_manuals__acme"
    assert beta._collection.name == "factory_floor_manuals__beta"

    acme_docs = build_retriever(acme, k=3, code_aware=True).invoke("F30021 ground fault")
    beta_docs = build_retriever(beta, k=3, code_aware=True).invoke("F30021 ground fault")

    assert acme_docs and all("ACME" in d.page_content for d in acme_docs)
    assert beta_docs and all("BETA" in d.page_content for d in beta_docs)
    assert not any("BETA" in d.page_content for d in acme_docs)


def test_default_tenant_uses_the_base_collection(tmp_path):
    default = _store(tmp_path, "default", "DEFAULT")
    assert default._collection.name == "factory_floor_manuals"
