import csv
from pathlib import Path

from factory_floor.config import HISTORY_CSV, MACHINES_CSV


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


def get_machine_history(machine_id: str, history_csv: Path = HISTORY_CSV) -> list:
    return [row for row in load_maintenance_history(history_csv) if row["machine_id"] == machine_id]
