"""Minimal FastAPI backend (phase 8 — proof, not the full surface).

Two endpoints, both thin wrappers over ``factory_floor.services`` — which is the whole
point of the phase-2 extraction: the same logic the Streamlit app calls can sit behind
an API with no duplication. The full intended surface (streaming SSE, resolutions,
audit, auth) is designed in ``docs/backend_architecture.md``, not built here.

Run:  uvicorn api.main:app --reload
"""

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from factory_floor import services
from factory_floor.config import get_settings

# Local-dev convenience: pick up a .env if present. In a real deployment the secret is
# injected into the environment by the runtime / vault and this is a no-op.
load_dotenv()
get_settings.cache_clear()

app = FastAPI(title="The Factory Floor API", version="0.1.0")

_state: dict = {}


def get_vectorstore():
    """One process-wide vector store, loaded lazily. Overridden in tests."""
    if "vs" not in _state:
        _state["vs"] = services.load_tenant_vectorstore(get_settings().tenant_id)
    return _state["vs"]


def get_llm():
    """None -> services.run_diagnostic builds the real model. Overridden in tests."""
    return None


class DiagnoseRequest(BaseModel):
    question: str = Field(..., min_length=1)
    machine_id: str = "GENERAL"
    equipment_type: str = ""
    language: str = "English"
    operator_id: str | None = None
    tenant_id: str = "default"


class DiagnoseResponse(BaseModel):
    answer: str | None
    sources: str | None
    tool_trace: list
    run_id: str | None
    safety_action: str | None
    cache_hit: bool
    blocked: bool
    cost_usd: float
    audit_id: int | None


@app.get("/health")
def health():
    return {"status": "ok", "tenant": get_settings().tenant_id}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(body: DiagnoseRequest, vectorstore=Depends(get_vectorstore), llm=Depends(get_llm)):
    req = services.DiagnosticRequest(
        question_text=body.question,
        machine_id=body.machine_id,
        equipment_type=body.equipment_type,
        language=body.language,
        operator_id=body.operator_id,
        tenant_id=body.tenant_id,
    )
    result = services.run_diagnostic(req, vectorstore=vectorstore, llm=llm)
    return DiagnoseResponse(
        answer=result.answer,
        sources=result.sources,
        tool_trace=result.tool_trace or [],
        run_id=result.run_id,
        safety_action=(result.safety or {}).get("action"),
        cache_hit=result.cache_hit,
        blocked=result.blocked,
        cost_usd=(result.cost or {}).get("total_usd", 0.0),
        audit_id=result.audit_id,
    )
