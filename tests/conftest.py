"""Shared pytest fixtures for The Factory Floor test suite.

Design notes:
- Unit tests never touch the network. The `_no_network` autouse fixture replaces the
  real `ChatOpenAI` / `OpenAIEmbeddings` constructors with something that raises, so a
  unit test that accidentally reaches for a real client fails loudly instead of hanging
  or billing.
- The whole point of the `llm=None` dependency-injection idiom throughout
  `factory_floor/` is to let tests pass a fake model. `make_fake_llm` / `make_structured_llm`
  build those fakes.
- `tmp_vectorstore` builds a genuine (tiny) Chroma store with deterministic fake
  embeddings, so integration tests exercise the real retriever / lexical-lookup code
  without an API key.
"""

import itertools
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

TESTS_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by directory so `-m unit` / `-m integration` work without every
    test file repeating a module-level marker."""
    for item in items:
        try:
            rel = Path(str(item.fspath)).relative_to(TESTS_ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "unit":
            item.add_marker(pytest.mark.unit)
        elif rel.parts and rel.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """factory_floor.config.get_settings() is lru_cached. Clear it around every test so
    monkeypatch.setenv on FACTORY_FLOOR_* actually takes effect."""
    from factory_floor.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Unit tests only: make constructing a real OpenAI client raise.

    Deliberately does NOT import anything — `langchain_openai` is slow to import in this
    environment (it pulls in transformers + torch), so a pure-logic test file must not
    pay that cost. It patches only the modules a test has already imported at module
    load time, which covers every realistic case (test files import their targets at
    the top).
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("llm"):
        return

    def _boom(*args, **kwargs):
        raise RuntimeError(
            "A unit test tried to construct a real LLM/embeddings client. "
            "Pass a fake via the llm= / embeddings= parameter, or mark the test @pytest.mark.integration."
        )

    for modname, attr in (
        ("langchain_openai", "ChatOpenAI"),
        ("langchain_openai", "OpenAIEmbeddings"),
        ("factory_floor.rag", "ChatOpenAI"),
        ("factory_floor.vectorstore", "OpenAIEmbeddings"),
    ):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, _boom, raising=False)


@pytest.fixture
def make_fake_llm():
    """Factory for a fake chat model. Pass the string(s) it should return; the last one
    repeats forever so a test never runs out of responses."""

    def _make(*responses):
        msgs = [AIMessage(content=r) if isinstance(r, str) else r for r in responses] or [
            AIMessage(content="Test answer.")
        ]
        stream = itertools.chain(msgs[:-1], itertools.repeat(msgs[-1]))
        return GenericFakeChatModel(messages=stream)

    return _make


@pytest.fixture
def fake_llm(make_fake_llm):
    return make_fake_llm("Test answer.")


@pytest.fixture
def make_structured_llm():
    """Factory for a fake model that supports `.with_structured_output(Schema)`.
    Pass the object(s) `.invoke()` should return in order (the last repeats)."""

    def _make(*values):
        seq = list(values) or [None]
        # One shared iterator across every .with_structured_output() call, so a caller
        # that re-creates the structured runnable per invocation (like audit_answers)
        # still walks the sequence in order rather than restarting it each time.
        shared_it = itertools.chain(seq[:-1], itertools.repeat(seq[-1]))

        class _Runnable:
            def invoke(self, *args, **kwargs):
                return next(shared_it)

        class _LLM:
            def with_structured_output(self, schema):
                return _Runnable()

            def invoke(self, *args, **kwargs):
                return AIMessage(content="(structured-fake plain invoke)")

        return _LLM()

    return _make


@pytest.fixture
def make_doc():
    """Factory for a LangChain Document with the metadata shape the project expects
    (`source_file`, `page`; extra keys via kwargs)."""

    def _make(content="chunk text", source_file="Manual.pdf", page=0, **extra):
        metadata = {"source_file": source_file, "page": page}
        metadata.update(extra)
        return Document(page_content=content, metadata=metadata)

    return _make


@pytest.fixture
def make_stub_retriever():
    """Factory for an object that duck-types `.invoke(query) -> [Document]` and records
    the queries it was asked."""

    def _make(docs):
        class _Stub:
            def __init__(self):
                self.calls = []

            def invoke(self, query):
                self.calls.append(query)
                return list(docs)

        return _Stub()

    return _make


@pytest.fixture
def tmp_vectorstore(tmp_path):
    """A real, tiny Chroma store with deterministic fake embeddings — no API key.
    Marked-integration tests use this to exercise the retriever and the lexical
    fault-code lookup for real."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    from factory_floor.vectorstore import build_vectorstore

    docs = [
        Document(
            page_content=(
                "F30021 Power unit: Ground fault. Cause: an earth fault between the power unit and "
                "the motor. Remedy: check the motor cable and the motor for a short-circuit to earth."
            ),
            metadata={"source_file": "Siemens_G120_CU240BE2_List_Manual.pdf", "page": 907, "equipment_type": "VFD"},
        ),
        Document(
            page_content=(
                "F30011 Power unit: Line phase failure in the main circuit. Check the line supply "
                "voltage and the main-circuit fuses."
            ),
            metadata={"source_file": "Siemens_G120_CU240BE2_List_Manual.pdf", "page": 900, "equipment_type": "VFD"},
        ),
        Document(
            page_content=(
                "Bearing maintenance: listen for abnormal noise, measure the insulation resistance, "
                "and re-grease per the lubricant plate interval."
            ),
            metadata={
                "source_file": "Siemens_SIMOTICS_SD_Operating_Instructions.pdf",
                "page": 40,
                "equipment_type": "electric_motor",
            },
        ),
        Document(
            page_content="See also: F30021 Note: this cross-reference is only relevant for chassis power units.",
            metadata={"source_file": "Siemens_G120_CU240BE2_List_Manual.pdf", "page": 61, "equipment_type": "VFD"},
        ),
    ]
    return build_vectorstore(
        docs,
        tmp_path / "vectorstore",
        "test_manuals",
        embeddings=DeterministicFakeEmbedding(size=1536),
        rebuild=True,
    )
