"""Semantic answer cache (phase 6).

When the same — or a very close — first-turn question comes back, serve the stored
answer instead of running the agent again. Opt-in
(``FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED=true``), same as ``rerank`` / ``code_aware``.

Scope and safety:
- Keyed by machine / equipment type / language / tenant, so a VFD answer never serves a
  motor question.
- **Fault-code questions** (``fault_codes.extract_codes`` non-empty) require an exact
  normalized-question match — never a semantic near-miss, so "F30021" is never served
  for "F30022". Prose questions use cosine similarity >= the configured threshold.
- Only first-turn, no-photo questions are cached, and only answers the safety gate let
  through (``pass`` / ``rewritten``, never ``held``).
- A ``version_stamp`` (embedding model + main collection name) and a TTL bound staleness;
  rebuilding the manual store should also call ``SemanticCache().clear()``.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document

from factory_floor.config import get_settings
from factory_floor.fault_codes import extract_codes, extract_possible_codes, normalize_code
from factory_floor.vectorstore import get_embeddings

CACHE_COLLECTION = "factory_floor_qa_cache"


def _normalize_question(text: str) -> str:
    return " ".join((text or "").lower().split())


def _code_signature(text: str) -> str:
    """Sorted, normalized fault codes in the question — '' when there are none. Also
    catches codes typed with display confusions ('F3OO21') so a code-bearing question is
    never treated as prose even if the typo pre-check upstream is skipped."""
    codes = {normalize_code(c) for c in extract_codes(text or "")}
    codes |= {norm for _typed, norm in extract_possible_codes(text or "")}
    return ",".join(sorted(codes))


@dataclass
class CacheHit:
    answer: str
    sources: str | None
    documents: list
    tool_trace: list
    run_id: str | None
    similarity: float
    cached_at: float


@dataclass
class SemanticCache:
    embeddings: object = None
    persist_directory: str = None
    _store: Chroma = field(default=None, repr=False)

    def __post_init__(self):
        settings = get_settings()
        self.persist_directory = str(self.persist_directory or settings.semantic_cache_dir)
        self.embeddings = self.embeddings or get_embeddings()
        self._store = Chroma(
            collection_name=CACHE_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},  # so relevance scores are real 0-1
        )

    # --- keys / stamps ------------------------------------------------------------

    @staticmethod
    def _version_stamp() -> str:
        s = get_settings()
        return f"{s.embedding_model}|{s.collection_name}"

    @staticmethod
    def _scope_filter(machine_id, equipment_type, language, tenant_id) -> dict:
        return {
            "$and": [
                {"machine_id": machine_id or "GENERAL"},
                {"equipment_type": equipment_type or ""},
                {"language": language or "English"},
                {"tenant_id": tenant_id or "default"},
            ]
        }

    @staticmethod
    def _entry_id(norm_q, machine_id, equipment_type, language, tenant_id) -> str:
        raw = "|".join([norm_q, machine_id or "GENERAL", equipment_type or "", language or "English",
                        tenant_id or "default"])
        return hashlib.sha256(raw.encode()).hexdigest()

    # --- read -------------------------------------------------------------------

    def lookup(self, question, *, machine_id="GENERAL", equipment_type="", language="English",
               tenant_id="default") -> CacheHit | None:
        norm_q = _normalize_question(question)
        if not norm_q:
            return None
        codes = _code_signature(question)
        threshold = get_settings().semantic_cache_similarity_threshold

        if codes:
            # exact match only — a metadata lookup, no embedding call
            got = self._store.get(
                where={"$and": [{"norm_q": norm_q}, {"codes": codes},
                                {"machine_id": machine_id or "GENERAL"},
                                {"equipment_type": equipment_type or ""},
                                {"language": language or "English"},
                                {"tenant_id": tenant_id or "default"}]},
                limit=1,
                include=["metadatas"],
            )
            metas = got.get("metadatas") or []
            if not metas:
                return None
            return self._hit_from_meta(metas[0], similarity=1.0)

        # Raw cosine distance -> similarity. The collection is created with
        # hnsw:space=cosine, so distance is in [0, 2] and similarity = 1 - distance.
        results = self._store.similarity_search_with_score(
            norm_q, k=1,
            filter=self._scope_filter(machine_id, equipment_type, language, tenant_id),
        )
        if not results:
            return None
        doc, distance = results[0]
        score = 1.0 - float(distance)
        if score < threshold:
            return None
        # a code-free question must not match a cached entry that *did* carry a code
        if (doc.metadata or {}).get("codes"):
            return None
        return self._hit_from_meta(doc.metadata, similarity=float(score))

    def _hit_from_meta(self, meta: dict, *, similarity: float) -> CacheHit | None:
        if meta.get("version_stamp") != self._version_stamp():
            return None
        ttl_hours = get_settings().semantic_cache_ttl_hours
        if ttl_hours and (time.time() - float(meta.get("cached_at", 0))) > ttl_hours * 3600:
            return None
        docs = [
            Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
            for d in json.loads(meta.get("documents_json", "[]"))
        ]
        return CacheHit(
            answer=meta.get("answer", ""),
            sources=meta.get("sources"),
            documents=docs,
            tool_trace=json.loads(meta.get("tool_trace_json", "[]")),
            run_id=meta.get("run_id"),
            similarity=similarity,
            cached_at=float(meta.get("cached_at", 0)),
        )

    # --- write ----------------------------------------------------------------

    def store(self, question, result, *, machine_id="GENERAL", equipment_type="",
              language="English", tenant_id="default") -> bool:
        """Cache one answer. Returns True if stored. Refuses held answers, blocked turns,
        and empty answers — the caller is responsible for only offering first-turn,
        no-photo questions."""
        safety_action = (result.safety or {}).get("action")
        if result.blocked or not (result.answer or "").strip():
            return False
        if safety_action not in ("pass", "rewritten"):
            return False

        norm_q = _normalize_question(question)
        documents_json = json.dumps([
            {"page_content": getattr(d, "page_content", ""), "metadata": dict(getattr(d, "metadata", {}))}
            for d in (result.documents or [])
        ])
        meta = {
            "norm_q": norm_q,
            "codes": _code_signature(question),
            "machine_id": machine_id or "GENERAL",
            "equipment_type": equipment_type or "",
            "language": language or "English",
            "tenant_id": tenant_id or "default",
            "answer": result.answer,
            "sources": result.sources or "",
            "documents_json": documents_json,
            "tool_trace_json": json.dumps(result.tool_trace or []),
            "run_id": result.run_id or "",
            "safety_action": safety_action,
            "cached_at": time.time(),
            "version_stamp": self._version_stamp(),
        }
        entry_id = self._entry_id(norm_q, machine_id, equipment_type, language, tenant_id)
        self._store.add_texts([norm_q], metadatas=[meta], ids=[entry_id])
        return True

    # --- maintenance --------------------------------------------------------------

    def count(self) -> int:
        return self._store._collection.count()

    def clear(self) -> None:
        self._store.delete_collection()
        self.__post_init__()
