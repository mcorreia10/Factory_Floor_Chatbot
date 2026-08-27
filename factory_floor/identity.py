"""Operator identity (phase 5).

An audit trail is only worth keeping if it records *who*. This is a deliberately small
identity layer: a committed ``operators.csv`` (same convention as ``machines.csv``) with
PBKDF2-hashed PINs, no external dependency. On a real shop floor this would be a badge
scan / PIN pad / SSO from the plant's MES — documented as the upgrade, not built here.
"""

import csv
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex

from factory_floor.config import OPERATORS_CSV

_PBKDF2_ROUNDS = 100_000
CSV_FIELDS = ["operator_id", "name", "role", "tenant_id", "salt", "pin_hash"]


@dataclass(frozen=True)
class Operator:
    operator_id: str
    name: str
    role: str
    tenant_id: str = "default"

    def as_dict(self) -> dict:
        return {
            "operator_id": self.operator_id,
            "name": self.name,
            "role": self.role,
            "tenant_id": self.tenant_id,
        }


def hash_pin(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", (pin or "").encode(), salt.encode(), _PBKDF2_ROUNDS).hex()


def new_salt() -> str:
    return token_hex(16)


def _rows(path=OPERATORS_CSV) -> list:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def list_operators(path=OPERATORS_CSV) -> list:
    return [Operator(r["operator_id"], r["name"], r["role"], r["tenant_id"]) for r in _rows(path)]


def authenticate(operator_id: str, pin: str, path=OPERATORS_CSV) -> Operator | None:
    """Return the Operator for a correct id + PIN, else None. Constant-time PIN compare."""
    for r in _rows(path):
        if r["operator_id"] != operator_id:
            continue
        if hmac.compare_digest(r["pin_hash"], hash_pin(pin, r["salt"])):
            return Operator(r["operator_id"], r["name"], r["role"], r["tenant_id"])
        return None
    return None
