"""Phase 8: the FastAPI proof, with the vector store and model overridden by fakes."""

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_llm, get_vectorstore

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_vectorstore, make_agent_fake_llm, monkeypatch):
    # gate-neutral answer so the default safety gate passes on the cheap keyword path
    monkeypatch.setenv("FACTORY_FLOOR_SAFETY_GATE_MODE", "off")
    app.dependency_overrides[get_vectorstore] = lambda: tmp_vectorstore
    app.dependency_overrides[get_llm] = lambda: make_agent_fake_llm("F30021 indicates a ground fault.")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_diagnose_returns_a_well_formed_result(client):
    r = client.post("/diagnose", json={"question": "what does F30021 mean?", "equipment_type": "VFD"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "F30021 indicates a ground fault."
    assert body["blocked"] is False
    assert body["cache_hit"] is False
    assert body["safety_action"] == "pass"
    assert set(body) == {
        "answer", "sources", "tool_trace", "run_id", "safety_action",
        "cache_hit", "blocked", "cost_usd", "audit_id",
    }


def test_diagnose_rejects_an_empty_question(client):
    r = client.post("/diagnose", json={"question": ""})
    assert r.status_code == 422  # pydantic min_length
