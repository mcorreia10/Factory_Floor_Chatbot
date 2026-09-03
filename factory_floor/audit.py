"""Audit trail + writable resolution history (phase 5).

A traceable record of every answer the copilot delivers — what was recommended, from
which manual pages, to whom, when, at what cost, and whether the safety gate touched it.
Plus a write path so an operator's actual resolution steps go back into the machine's
history.

Storage is stdlib ``sqlite3`` with ``journal_mode=WAL`` — this is the answer to the
concurrent-write race flagged in ``docs/limitations_and_opportunities.md`` #6: several
Streamlit sessions can write at once without "database is locked". The static
``maintenance_history.csv`` stays read-only; live events live here and the UI unions the
two.

This module also owns ``DailyLedger`` (re-exported from ``cost.py`` for backwards
compatibility) — the phase-3 JSONL ledger folded into the ``cost_events`` table.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from factory_floor.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    operator_id TEXT,
    machine_id TEXT,
    question TEXT,
    standalone_question TEXT,
    delivered_answer TEXT,
    original_answer TEXT,
    safety_action TEXT,
    safety_reason TEXT,
    cache_hit INTEGER DEFAULT 0,
    run_id TEXT,
    language TEXT,
    cost_usd REAL DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS recommendation_sources (
    recommendation_id INTEGER NOT NULL,
    ordinal INTEGER,
    source_file TEXT,
    page INTEGER,
    equipment_type TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    recommendation_id INTEGER NOT NULL,
    ordinal INTEGER,
    tool TEXT,
    tool_input TEXT,
    output_preview TEXT
);
CREATE TABLE IF NOT EXISTS cost_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    date TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    usd REAL DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    n_calls INTEGER DEFAULT 0,
    recommendation_id INTEGER
);
CREATE TABLE IF NOT EXISTS resolution_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER,
    machine_id TEXT NOT NULL,
    operator_id TEXT,
    ts_utc TEXT NOT NULL,
    steps_text TEXT NOT NULL,
    cmms_exported_at TEXT,
    fault_code TEXT,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS ix_cost_events_day ON cost_events(date, tenant_id);
CREATE INDEX IF NOT EXISTS ix_resolution_machine ON resolution_events(machine_id);
"""

# Indexes over columns that _ADDED_COLUMNS introduces. These cannot live in _SCHEMA: on a
# database created before those columns existed, executescript() runs before the ALTERs
# and would fail with "no such column".
_SCHEMA_AFTER_MIGRATION = """
CREATE INDEX IF NOT EXISTS ix_resolution_code ON resolution_events(machine_id, fault_code);
"""

# Columns added after the table shipped. CREATE TABLE IF NOT EXISTS will not add them to
# an existing database, so init_db() applies these too — keyed on the live PRAGMA rather
# than a version number, which keeps it idempotent and needs no migration bookkeeping.
_ADDED_COLUMNS = {
    "resolution_events": {
        # Which fault the operator was resolving. Without it a recorded resolution could
        # not be matched to a later occurrence of the same code — the blocker that made
        # the zero-cost prior-occurrence panel impossible to build.
        "fault_code": "TEXT",
        # Whether it actually worked: resolved | temporary | not_resolved | "" (legacy,
        # unknown). "Most frequent action" is not "most effective action" without this —
        # VFD-04's power unit module was replaced twice and F30805 still came back.
        "outcome": "TEXT",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def db_path(path=None) -> Path:
    return Path(path) if path else get_settings().audit_db_path


@contextmanager
def _connect(path=None):
    p = db_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path=None) -> None:
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
        for table, columns in _ADDED_COLUMNS.items():
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        conn.executescript(_SCHEMA_AFTER_MIGRATION)


# --- recommendations ----------------------------------------------------------------

def record_recommendation(result, req, path=None) -> int:
    """Persist one delivered answer + its sources + tool calls + a linked cost row.
    ``result`` is a services.DiagnosticResult, ``req`` a services.DiagnosticRequest."""
    init_db(path)
    cost = result.cost or {}
    safety = result.safety or {}
    with _connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO recommendations
               (ts_utc, tenant_id, operator_id, machine_id, question, standalone_question,
                delivered_answer, original_answer, safety_action, safety_reason, cache_hit,
                run_id, language, cost_usd, input_tokens, output_tokens)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), req.tenant_id, req.operator_id, req.machine_id, req.question_text,
                None, result.answer, safety.get("original_answer"),
                safety.get("action"), safety.get("reason"),
                1 if result.cache_hit else 0, result.run_id, result.language,
                cost.get("total_usd", 0.0), cost.get("total_input_tokens", 0),
                cost.get("total_output_tokens", 0),
            ),
        )
        rec_id = int(cur.lastrowid)

        ordinal = 0
        for doc in result.documents or []:
            md = getattr(doc, "metadata", {}) or {}
            if md.get("not_found"):
                continue  # the code-not-found notice is an instruction, not a citable source
            ordinal += 1
            conn.execute(
                "INSERT INTO recommendation_sources "
                "(recommendation_id, ordinal, source_file, page, equipment_type) VALUES (?,?,?,?,?)",
                (rec_id, ordinal, md.get("source_file"), (md.get("page", 0) or 0) + 1, md.get("equipment_type")),
            )

        for i, entry in enumerate(result.tool_trace or [], 1):
            conn.execute(
                "INSERT INTO tool_calls "
                "(recommendation_id, ordinal, tool, tool_input, output_preview) VALUES (?,?,?,?,?)",
                (rec_id, i, entry.get("tool"), json.dumps(entry.get("input")),
                 (entry.get("output_preview") or "")[:300]),
            )

        if cost.get("n_calls"):
            conn.execute(
                "INSERT INTO cost_events "
                "(ts_utc, date, tenant_id, usd, input_tokens, output_tokens, n_calls, recommendation_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (_now(), _today(), req.tenant_id, cost.get("total_usd", 0.0),
                 cost.get("total_input_tokens", 0), cost.get("total_output_tokens", 0),
                 cost.get("n_calls", 0), rec_id),
            )
    return rec_id


def get_audit_trail(machine_id=None, operator_id=None, tenant_id=None, limit=100, path=None) -> list:
    init_db(path)
    clauses, params = [], []
    for column, value in (("machine_id", machine_id), ("operator_id", operator_id), ("tenant_id", tenant_id)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _connect(path) as conn:
        rows = conn.execute(
            f"SELECT * FROM recommendations {where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
    return [dict(r) for r in rows]


# --- resolution events (writable history) ------------------------------------------

def append_resolution_event(recommendation_id, *, machine_id, operator_id, steps_text,
                            fault_code=None, outcome=None, path=None) -> int:
    """``fault_code`` is what makes a resolution findable again on the next occurrence;
    ``outcome`` (resolved | temporary | not_resolved) is what separates "done most often"
    from "actually worked". Both default to "" so pre-existing callers keep working."""
    init_db(path)
    with _connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO resolution_events "
            "(recommendation_id, machine_id, operator_id, ts_utc, steps_text, fault_code, outcome) "
            "VALUES (?,?,?,?,?,?,?)",
            (recommendation_id, machine_id, operator_id, _now(), steps_text,
             (fault_code or "").strip().upper(), (outcome or "").strip()),
        )
    return int(cur.lastrowid)


def get_resolution_events(machine_id, path=None) -> list:
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM resolution_events WHERE machine_id = ? ORDER BY id", (machine_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def export_to_cmms(resolution_event_id, path=None, outbox=None) -> dict:
    """Demo of the CMMS / ERP integration point (Opportunity #2). Appends the resolution
    record to a local outbox file and stamps the event. A real integration replaces the
    file append with an authenticated POST to the CMMS — e.g. SAP PM
    ``/sap/opu/odata/sap/MAINTENANCEORDER`` or IBM Maximo ``/oslc/os/mxwo`` — or a
    webhook; the payload below maps 1:1 to a work-order "actuals" record."""
    init_db(path)
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM resolution_events WHERE id = ?", (resolution_event_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no resolution_event with id {resolution_event_id}")
        stamp = _now()
        conn.execute(
            "UPDATE resolution_events SET cmms_exported_at = ? WHERE id = ?", (stamp, resolution_event_id)
        )

    payload = {
        "exported_at": stamp,
        "resolution_event_id": resolution_event_id,
        "machine_id": row["machine_id"],
        "operator_id": row["operator_id"],
        "steps": row["steps_text"],
    }
    outbox_path = Path(outbox) if outbox else get_settings().cmms_outbox_path
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    with outbox_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    return {"status": "accepted", "cmms_ref": f"DEMO-{resolution_event_id:06d}", "exported_at": stamp}


# --- daily spend ledger (folded in from cost.py) ----------------------------------

class DailyLedger:
    """Per-turn cost rows in the ``cost_events`` table. Same ``record`` / ``today_total``
    API the phase-3 JSONL version had; ``services.run_diagnostic`` uses ``today_total``
    for the pre-flight spend cap. When the audit trail is enabled the cost row is written
    by ``record_recommendation`` (linked to the recommendation); when it's disabled,
    ``DailyLedger.record`` writes a standalone row."""

    def __init__(self, db_path=None):
        self.db_path = db_path

    def record(self, *, tenant_id, usage) -> None:
        init_db(self.db_path)
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO cost_events "
                "(ts_utc, date, tenant_id, usd, input_tokens, output_tokens, n_calls) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), _today(), tenant_id, usage.total_usd,
                 usage.total_input_tokens, usage.total_output_tokens, usage.n_calls),
            )

    def today_total(self, tenant_id: str = "default") -> float:
        if not db_path(self.db_path).exists():
            return 0.0
        init_db(self.db_path)
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(usd), 0) AS total FROM cost_events WHERE date = ? AND tenant_id = ?",
                (_today(), tenant_id),
            ).fetchone()
        return float(row["total"])
