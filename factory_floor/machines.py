import csv
from pathlib import Path

from factory_floor.config import HISTORY_CSV, MACHINES_CSV

# What an operator can say about a resolution they recorded. "temporary" exists because
# it is the honest answer more often than either extreme — VFD-04's F30805 came back
# after two power-unit replacements — and because a tally that cannot tell "held" from
# "came back next week" cannot rank actions by effectiveness.
OUTCOME_LABELS = {
    "resolved": "resolved",
    "temporary": "worked temporarily, fault returned",
    "not_resolved": "did not resolve",
}


def load_machines(machines_csv: Path = MACHINES_CSV) -> list:
    with Path(machines_csv).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_maintenance_history(history_csv: Path = HISTORY_CSV) -> list:
    with Path(history_csv).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_machine(machine_id: str, machines_csv: Path = MACHINES_CSV) -> dict | None:
    """Single machine's registry row, or None for an unknown id or the UI-only
    'GENERAL' sentinel (which is not in machines.csv)."""
    return next((row for row in load_machines(machines_csv) if row["machine_id"] == machine_id), None)


def get_machine_history(machine_id: str, history_csv: Path = HISTORY_CSV,
                        include_resolutions: bool = False) -> list:
    """The machine's events. The static ``maintenance_history.csv`` is always included
    and stays read-only. With ``include_resolutions=True`` the live operator-recorded
    resolution events (SQLite, phase 5) are unioned in, each tagged by origin — the
    default is False so notebook 05 and the agent's history tool are unchanged."""
    rows = [row for row in load_maintenance_history(history_csv) if row["machine_id"] == machine_id]
    if include_resolutions:
        from factory_floor import audit

        for event in audit.get_resolution_events(machine_id):
            outcome = (event.get("outcome") or "").strip()
            rows.append(
                {
                    "event_id": f"R{event['id']}",
                    "machine_id": machine_id,
                    "event_date": (event["ts_utc"] or "")[:10],
                    "event_type": "operator_resolution",
                    "fault_code": (event.get("fault_code") or "").strip(),
                    "description": (
                        f"Resolution recorded via the copilot by {event['operator_id'] or 'unknown'}"
                        + (f" — outcome: {OUTCOME_LABELS.get(outcome, outcome)}" if outcome else "")
                    ),
                    "action_taken": event["steps_text"],
                    "technician": event["operator_id"] or "",
                    "downtime_hours": "",
                    "outcome": outcome,
                }
            )
    return rows


def append_resolution_event(machine_id: str, *, operator_id: str, steps_text: str,
                            recommendation_id: int | None = None,
                            fault_code: str | None = None, outcome: str | None = None) -> int:
    """Write an operator's actual resolution steps back into the machine's history
    (SQLite). Returns the new resolution_event id.

    ``fault_code`` makes the resolution findable on the next occurrence of the same
    fault; ``outcome`` records whether it actually held."""
    from factory_floor import audit

    return audit.append_resolution_event(
        recommendation_id, machine_id=machine_id, operator_id=operator_id,
        steps_text=steps_text, fault_code=fault_code, outcome=outcome,
    )
