"""Multi-tenant seam (phase 7 — design-only + seams).

This build runs a single tenant, ``default``. The seam is here so a real multi-tenant
deployment routes each request to its own manual collection without touching business
logic: ``resolve_collection(tenant_id)`` is called wherever a collection name is needed
(``services.load_tenant_vectorstore``, the ingestion notebooks), and ``tenant_id`` is
already carried on every request, audit row, cost row, and cache key from the earlier
phases.

See ``docs/multi_tenancy.md`` for the full design.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from factory_floor.config import TENANTS_CSV, get_settings

DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    name: str
    vector_collection: str
    created_at: str = ""


def resolve_collection(tenant_id: str | None) -> str:
    """The Chroma collection name for a tenant. ``default`` (or unset) keeps the existing
    ``settings.collection_name`` so the current single-tenant store is unchanged; any
    other tenant gets a suffixed collection that never overlaps."""
    base = get_settings().collection_name
    if not tenant_id or tenant_id == DEFAULT_TENANT:
        return base
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in tenant_id)
    return f"{base}__{safe}"


def list_tenants(path=TENANTS_CSV) -> list:
    p = Path(path)
    if not p.exists():
        return [Tenant(DEFAULT_TENANT, "Default", get_settings().collection_name)]
    with p.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Tenant(
            r["tenant_id"],
            r.get("name", r["tenant_id"]),
            r.get("vector_collection") or resolve_collection(r["tenant_id"]),
            r.get("created_at", ""),
        )
        for r in rows
    ]


def get_tenant(tenant_id: str, path=TENANTS_CSV) -> Tenant | None:
    return next((t for t in list_tenants(path) if t.tenant_id == tenant_id), None)
