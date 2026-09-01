"""Prior occurrences of a fault on one machine — computed, never generated.

There is no LLM in this module, on purpose. Everything here is counting, sorting and
date arithmetic over records that already exist, so the panel it feeds is free to
produce and cannot hallucinate: every line traces to a row in maintenance_history.csv
or resolution_events.

What this deliberately does NOT do is recommend an action. The obvious next step is
"rank the past actions and suggest the most common one", and on the real data that
advice would be actively wrong: VFD-04's F30805 was answered twice with "Replaced the
power unit module" and still came back, then three times with a power cycle. A tally
would recommend the power cycle 3-to-2 — i.e. recommend masking a recurring hardware
fault. Ranking by effectiveness needs the ``outcome`` field to have been filled in for
long enough to mean something; until then this module reports and lets the operator
(or the agent, if they ask for a diagnosis) decide.
"""

from datetime import date

from factory_floor.fault_codes import extract_codes, normalize_code
from factory_floor.machines import get_machine_history

# Below this many days between two occurrences of the same code, the earlier fix did not
# hold in any practical sense. Four weeks is a maintenance-cycle judgement, not a
# measured threshold — it is a flag for the operator to read, never an automated verdict.
RECURRENCE_WINDOW_DAYS = 28


def code_in_question(text: str) -> str | None:
    """The single fault code a question is about, or None. Returns None when the text
    names more than one code — the panel answers "has THIS fault happened before?", and
    with two codes there is no single answer to show."""
    codes = {normalize_code(c) for c in extract_codes(text or "")}
    return next(iter(codes)) if len(codes) == 1 else None


def _parse(day: str):
    try:
        return date.fromisoformat((day or "")[:10])
    except ValueError:
        return None


def find_prior_occurrences(machine_id: str, fault_code: str) -> list:
    """Every past event for this machine that carries this fault code — static history
    and operator-recorded resolutions alike — oldest first."""
    if not machine_id or machine_id == "GENERAL" or not fault_code:
        return []
    wanted = normalize_code(fault_code)
    rows = [
        row for row in get_machine_history(machine_id, include_resolutions=True)
        if normalize_code(row.get("fault_code") or "") == wanted
    ]
    return sorted(rows, key=lambda r: r.get("event_date") or "")


def summarize(occurrences: list) -> dict:
    """Facts about a list of prior occurrences. Empty-safe."""
    if not occurrences:
        return {"count": 0, "recurring": False, "outcomes": {}, "actions": [],
                "first_date": None, "last_date": None, "shortest_gap_days": None}

    dates = [d for d in (_parse(r.get("event_date")) for r in occurrences) if d]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])] if len(dates) > 1 else []
    outcomes: dict = {}
    for row in occurrences:
        key = (row.get("outcome") or "").strip()
        if key:
            outcomes[key] = outcomes.get(key, 0) + 1

    return {
        "count": len(occurrences),
        # "Recurring" means it came back at all, not that it came back quickly — two
        # occurrences four years apart is still a fault that returned after a repair.
        "recurring": len(occurrences) > 1,
        "shortest_gap_days": min(gaps) if gaps else None,
        "returned_quickly": bool(gaps) and min(gaps) <= RECURRENCE_WINDOW_DAYS,
        "first_date": dates[0].isoformat() if dates else None,
        "last_date": dates[-1].isoformat() if dates else None,
        "outcomes": outcomes,
        "actions": [
            {
                "date": row.get("event_date") or "",
                "action": (row.get("action_taken") or "").strip(),
                "who": (row.get("technician") or "").strip(),
                "source": "operator" if row.get("event_type") == "operator_resolution" else "logbook",
                "outcome": (row.get("outcome") or "").strip(),
                "downtime_hours": row.get("downtime_hours") or "",
            }
            for row in occurrences
        ],
    }


def prior_occurrence_report(machine_id: str, question_text: str) -> dict | None:
    """The whole zero-cost check for one submitted question: is there a single fault code
    in it, and has that code been seen on this machine before? Returns None when there is
    nothing worth interrupting the operator for — no code, no machine, or no history —
    in which case the caller just runs the normal diagnosis."""
    code = code_in_question(question_text)
    if not code:
        return None
    occurrences = find_prior_occurrences(machine_id, code)
    if not occurrences:
        return None
    report = summarize(occurrences)
    report["fault_code"] = code
    report["machine_id"] = machine_id
    return report
