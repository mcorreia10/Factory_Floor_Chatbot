"""Application service layer — the business logic of a diagnostic turn, with no
Streamlit dependency.

Before this, ``app.py::submit_turn`` did everything inline: typo pre-check, photo
classification, retriever construction, agent call, and turn assembly, all tangled with
``st.*`` calls. That made it impossible to test the flow, reuse it from a notebook, or
put an API in front of it.

Everything here is a plain function or dataclass. ``run_diagnostic`` is the single seam
the later phases wrap (cost metering, the blocking safety gate, the audit record, the
semantic cache) — which is why it comes before all of them.
"""

import io
from dataclasses import dataclass, field
from typing import Any

from factory_floor import audit
from factory_floor.agent import run_diagnostic_agent, stream_diagnostic_agent
from factory_floor.config import get_settings
from factory_floor.cost import (
    PENDING_TURN_ESTIMATE_USD,
    CostTrackingCallback,
    DailyLedger,
    SpendCapExceeded,
    UsageAccumulator,
    check_spend_cap,
)
from factory_floor.fault_codes import extract_possible_codes
from factory_floor.rag import build_chat_history, build_retriever, get_llm
from factory_floor.safety import enforce_safety
from factory_floor.vision import classify_defect_trained

DEFAULT_TOP_K = 5

# Re-exported so app.py / notebooks build chat history through the service layer rather
# than reaching into rag.py directly.
__all__ = [
    "DiagnosticRequest",
    "DiagnosticResult",
    "build_chat_history",
    "check_typo",
    "classify_photo",
    "build_diagnostic_retriever",
    "run_diagnostic",
    "assemble_turn",
]


@dataclass
class DiagnosticRequest:
    """Everything one diagnostic turn needs, gathered by the caller (the UI reads it off
    widgets + session state; an API would read it off the request body)."""

    question_text: str
    machine_id: str
    equipment_type: str = ""
    chat_history: list = field(default_factory=list)
    vision_context: str | None = None
    language: str = "English"
    operator_id: str | None = None
    tenant_id: str = "default"


@dataclass
class DiagnosticResult:
    """The outcome of a turn. The trailing fields stay at their defaults in phase 2 and
    get populated as later phases land (cost, safety, audit_id, cache_hit, blocked) —
    kept here now so the dataclass shape is stable."""

    question: str
    answer: str | None
    documents: list
    sources: str | None
    tool_trace: list | None
    run_id: str | None
    language: str
    cost: dict | None = None
    safety: dict | None = None
    audit_id: int | None = None
    cache_hit: bool = False
    blocked: bool = False
    message: str | None = None


def check_typo(question_text: str) -> list[tuple[str, str]]:
    """(as_typed, suggestion) pairs for fault codes typed with character confusions that
    map onto a real code — e.g. ('F3OO21', 'F30021'). The caller decides how to ask the
    operator which they meant; nothing is auto-corrected."""
    return extract_possible_codes(question_text or "")


def classify_photo(image_bytes: bytes, clf) -> dict | None:
    """Run the trained defect classifier on an uploaded photo and build the text summary
    the agent will see. Returns ``None`` when no classifier is available (same
    short-circuit as before). ``clf`` is the loaded classifier tuple ``(model, labels)``
    passed straight to ``classify_defect_trained``."""
    if clf is None:
        return None
    classification = classify_defect_trained(io.BytesIO(image_bytes), clf)
    vision_context = (
        "Vision analysis of the uploaded photo: predicted condition = "
        f"{classification['predicted_label']} (confidence {classification['confidence']:.0%}, "
        f"{'defective' if classification['is_defective'] else 'no defect detected'})."
    )
    return {"classification": classification, "vision_context": vision_context}


def build_diagnostic_retriever(vectorstore, equipment_type: str, llm, *, k: int = DEFAULT_TOP_K,
                               tenant_id: str = "default"):
    """The retriever the diagnostic agent uses — reranked + code-aware, filtered to the
    selected machine's equipment type. ``tenant_id`` is a no-op placeholder now; phase 7
    routes it to a per-tenant collection."""
    return build_retriever(
        vectorstore,
        k=k,
        equipment_type=equipment_type or None,
        rerank=True,
        rerank_llm=llm,
        code_aware=True,
    )


def _blocked_result(req: DiagnosticRequest, message: str) -> DiagnosticResult:
    return DiagnosticResult(
        question=req.question_text,
        answer=None,
        documents=[],
        sources=None,
        tool_trace=None,
        run_id=None,
        language=req.language,
        blocked=True,
        message=message,
    )


def run_diagnostic(req: DiagnosticRequest, *, vectorstore, llm=None, stream: bool = False):
    """Run one diagnostic turn.

    ``stream=False`` -> returns a finished ``DiagnosticResult``.
    ``stream=True``  -> returns ``(generator, DiagnosticResult)``; the result's answer/
    documents/sources/tool_trace/cost stay ``None``/empty until the caller exhausts the
    generator. A turn blocked by the spend cap still returns ``(empty_generator,
    result)`` in streaming mode, with ``result.blocked`` set — the caller checks that
    before consuming.

    Phase 3 wraps the agent call with a cost-tracking callback and a pre-flight daily
    spend cap. Phases 4-6 wrap this further.
    """
    settings = get_settings()
    accumulator = UsageAccumulator()
    callback = CostTrackingCallback(accumulator, default_model=settings.llm_model)
    ledger = DailyLedger()  # -> the audit SQLite cost_events table (phase 5)

    if settings.daily_spend_cap_usd is not None:
        try:
            check_spend_cap(
                ledger.today_total(req.tenant_id),
                cap=settings.daily_spend_cap_usd,
                pending_estimate=PENDING_TURN_ESTIMATE_USD,
                alert_threshold=settings.cost_alert_threshold,
            )
        except SpendCapExceeded as exc:
            blocked = _blocked_result(req, str(exc))
            return (iter(()), blocked) if stream else blocked

    llm = llm or get_llm()
    retriever = build_diagnostic_retriever(
        vectorstore, req.equipment_type, llm, tenant_id=req.tenant_id
    )
    agent_config = {"callbacks": [callback]}
    gate_active = settings.safety_gate_mode != "off"

    def _apply_gate(raw_answer: str | None):
        gate = enforce_safety(
            raw_answer, llm=llm, mode=settings.safety_gate_mode, language=req.language
        )
        return gate.delivered_answer, gate.as_dict()

    def _finalize(result: DiagnosticResult) -> None:
        result.cost = accumulator.as_dict()
        if settings.audit_enabled:
            # record_recommendation also writes the linked cost_events row.
            result.audit_id = audit.record_recommendation(result, req)
        elif accumulator.n_calls:
            ledger.record(tenant_id=req.tenant_id, usage=accumulator)

    if stream:
        generator, streamed = stream_diagnostic_agent(
            req.question_text,
            retriever,
            req.machine_id,
            chat_history=req.chat_history,
            vision_context=req.vision_context,
            llm=llm,
            language=req.language,
            config=agent_config,
        )
        result = DiagnosticResult(
            question=req.question_text,
            answer=None,
            documents=[],
            sources=None,
            tool_trace=None,
            run_id=streamed.run_id,
            language=req.language,
        )

        def _wrapped():
            # With the gate active the answer can change (rewrite) or be withheld
            # (hold), so the raw tokens can't be shown live — drain silently, gate the
            # finished text, then emit it once. With the gate off, stream normally.
            if gate_active:
                for _ in generator:
                    pass
            else:
                yield from generator
            result.documents = streamed.documents
            result.sources = streamed.sources
            result.tool_trace = streamed.tool_trace
            result.answer, result.safety = _apply_gate(streamed.answer)
            _finalize(result)
            if gate_active:
                yield result.answer or ""

        return _wrapped(), result

    raw = run_diagnostic_agent(
        req.question_text,
        retriever,
        req.machine_id,
        chat_history=req.chat_history,
        vision_context=req.vision_context,
        llm=llm,
        language=req.language,
        config=agent_config,
    )
    delivered, safety = _apply_gate(raw["answer"])
    result = DiagnosticResult(
        question=raw["question"],
        answer=delivered,
        documents=raw["documents"],
        sources=raw["sources"],
        tool_trace=raw["tool_trace"],
        run_id=raw["run_id"],
        language=raw["language"],
        safety=safety,
    )
    _finalize(result)
    return result


def assemble_turn(result: DiagnosticResult, *, image_bytes: bytes | None,
                  classification: dict | None, vision_context: str | None,
                  language: str) -> dict[str, Any]:
    """Build the per-turn dict the UI appends to its conversation list. Identical shape
    to what ``app.py`` produced inline before the extraction."""
    return {
        "type": "agent",
        "question": result.question or "[Uploaded a photo of a component]",
        "answer": result.answer,
        "documents": result.documents,
        "sources": result.sources,
        "tool_trace": result.tool_trace,
        "safety": result.safety,
        "audit_id": result.audit_id,
        "image_bytes": image_bytes,
        "vision_context": vision_context,
        "predicted_label": classification["predicted_label"] if classification else None,
        "is_defective": classification["is_defective"] if classification else None,
        "language": language,
    }
