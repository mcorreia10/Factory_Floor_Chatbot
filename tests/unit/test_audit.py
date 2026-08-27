"""factory_floor/audit.py (phase 5) — the SQLite audit trail + writable resolutions.

The conftest isolation fixture points FACTORY_FLOOR_AUDIT_DB_PATH / _CMMS_OUTBOX_PATH at
tmp_path, so `path=None` calls stay off the real files; tests also pass explicit paths
for clarity where it matters.
"""

import json
import sqlite3
from types import SimpleNamespace

from langchain_core.documents import Document

from factory_floor import audit


def _db(tmp_path):
    return tmp_path / "audit.sqlite3"


def _result(**over):
    base = dict(
        answer="Isolate first, then check the motor cable.",
        documents=[
            Document(page_content="def", metadata={"source_file": "LM.pdf", "page": 907, "equipment_type": "VFD"}),
            Document(page_content="notice", metadata={"source_file": "(not in corpus)", "page": -1, "not_found": True}),
        ],
        sources="- LM.pdf — page 908",
        tool_trace=[{"tool": "search_manuals", "input": {"query": "F30021"}, "output_preview": "..."}],
        run_id="run-1",
        language="English",
        cost={"total_usd": 0.0021, "total_input_tokens": 2900, "total_output_tokens": 300, "n_calls": 2, "by_model": {}},
        safety={"action": "rewritten", "reason": "added precautions", "original_answer": "check the motor cable"},
        cache_hit=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _req(**over):
    base = dict(question_text="F30021 ground fault?", machine_id="VFD-06", operator_id="OP-1001", tenant_id="default")
    base.update(over)
    return SimpleNamespace(**base)


class TestInitDb:
    def test_creates_every_table(self, tmp_path):
        audit.init_db(_db(tmp_path))
        with sqlite3.connect(_db(tmp_path)) as conn:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"recommendations", "recommendation_sources", "tool_calls", "cost_events", "resolution_events"} <= names

    def test_idempotent(self, tmp_path):
        audit.init_db(_db(tmp_path))
        audit.init_db(_db(tmp_path))  # must not raise


class TestRecordRecommendation:
    def test_roundtrip_with_sources_tools_and_cost(self, tmp_path):
        rec_id = audit.record_recommendation(_result(), _req(), path=_db(tmp_path))
        trail = audit.get_audit_trail(path=_db(tmp_path))
        assert len(trail) == 1
        row = trail[0]
        assert row["id"] == rec_id
        assert row["operator_id"] == "OP-1001"
        assert row["machine_id"] == "VFD-06"
        assert row["safety_action"] == "rewritten"
        assert row["original_answer"] == "check the motor cable"
        assert row["cost_usd"] > 0

        with sqlite3.connect(_db(tmp_path)) as conn:
            conn.row_factory = sqlite3.Row
            srcs = conn.execute("SELECT * FROM recommendation_sources WHERE recommendation_id=?", (rec_id,)).fetchall()
            tools = conn.execute("SELECT * FROM tool_calls WHERE recommendation_id=?", (rec_id,)).fetchall()
            costs = conn.execute("SELECT * FROM cost_events WHERE recommendation_id=?", (rec_id,)).fetchall()
        assert len(srcs) == 1  # the not_found doc is skipped
        assert srcs[0]["page"] == 908
        assert len(tools) == 1 and tools[0]["tool"] == "search_manuals"
        assert json.loads(tools[0]["tool_input"]) == {"query": "F30021"}
        assert len(costs) == 1 and costs[0]["n_calls"] == 2

    def test_cache_hit_flag_is_persisted(self, tmp_path):
        audit.record_recommendation(_result(cache_hit=True), _req(), path=_db(tmp_path))
        assert audit.get_audit_trail(path=_db(tmp_path))[0]["cache_hit"] == 1


class TestGetAuditTrail:
    def test_filters(self, tmp_path):
        audit.record_recommendation(_result(), _req(machine_id="VFD-06", operator_id="OP-1001"), path=_db(tmp_path))
        audit.record_recommendation(_result(), _req(machine_id="MOTOR-02", operator_id="OP-1002"), path=_db(tmp_path))
        assert len(audit.get_audit_trail(machine_id="VFD-06", path=_db(tmp_path))) == 1
        assert len(audit.get_audit_trail(operator_id="OP-1002", path=_db(tmp_path))) == 1
        assert len(audit.get_audit_trail(path=_db(tmp_path))) == 2


class TestResolutionEvents:
    def test_append_and_read_back(self, tmp_path):
        ev_id = audit.append_resolution_event(
            42, machine_id="VFD-06", operator_id="OP-1001", steps_text="Isolated, replaced cable, retested.",
            path=_db(tmp_path),
        )
        events = audit.get_resolution_events("VFD-06", path=_db(tmp_path))
        assert len(events) == 1
        assert events[0]["id"] == ev_id
        assert events[0]["recommendation_id"] == 42
        assert events[0]["steps_text"].startswith("Isolated")
        assert events[0]["cmms_exported_at"] is None

    def test_export_to_cmms_writes_outbox_and_stamps(self, tmp_path):
        ev_id = audit.append_resolution_event(
            None, machine_id="VFD-06", operator_id="OP-1001", steps_text="Re-torqued the terminals.",
            path=_db(tmp_path),
        )
        outbox = tmp_path / "outbox.jsonl"
        ack = audit.export_to_cmms(ev_id, path=_db(tmp_path), outbox=outbox)

        assert ack["status"] == "accepted"
        assert ack["cmms_ref"] == f"DEMO-{ev_id:06d}"
        line = json.loads(outbox.read_text().splitlines()[0])
        assert line["machine_id"] == "VFD-06"
        assert line["steps"] == "Re-torqued the terminals."
        assert audit.get_resolution_events("VFD-06", path=_db(tmp_path))[0]["cmms_exported_at"] is not None

    def test_export_unknown_event_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            audit.export_to_cmms(999, path=_db(tmp_path), outbox=tmp_path / "o.jsonl")
