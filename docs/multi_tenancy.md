# Multi-tenancy

**Status:** design + seams. This build runs a single tenant, `default`. Everything below
is threaded through the code so a real multi-tenant deployment is a configuration and
data exercise, not a rewrite — but the running app is single-tenant.

## The model

| Concept | Identifies | Isolation it buys |
|---|---|---|
| **Tenant** | a plant / customer | its own manual corpus, its own audit + cost data, its own spend cap |
| **Operator** | a person (`operator_id`) | the "to whom" of every audit row; maps to exactly one tenant |

`tenants.csv` (root, committed) is the registry — `tenant_id, name, vector_collection,
created_at`. Today it has one row, `default`.

## What is already threaded

- **`tenant_id` on every request** — `services.DiagnosticRequest.tenant_id`; the app
  reads it from the signed-in operator (`operator["tenant_id"]`), default `"default"`.
- **Per-tenant manual collection** — `factory_floor/tenancy.py::resolve_collection(tenant_id)`
  returns `settings.collection_name` for `default` and `"{base}__{tenant_id}"` otherwise.
  `services.load_tenant_vectorstore(tenant_id)` uses it; `app.py::load_rag_components`
  is keyed on `tenant_id` (its `@st.cache_resource` cache key).
- **Per-tenant audit + cost** — the `recommendations`, `cost_events` and
  `resolution_events` tables all carry `tenant_id`; `get_audit_trail(tenant_id=…)` and
  `DailyLedger.today_total(tenant_id)` filter on it, so the daily spend cap is per
  tenant.
- **Per-tenant answer cache** — `SemanticCache` scopes every key by `tenant_id`, so one
  tenant's cached answer can never be served to another.
- **Per-tenant operators** — `operators.csv` has a `tenant_id` column; `authenticate`
  returns it on the `Operator`.

## What a real deployment still needs to build

1. **Ingestion parametrized by tenant.** Notebooks 01–02 build the `default` collection.
   For a tenant `acme`, they need one change: read `TENANT_ID` from the environment and
   pass `resolve_collection(TENANT_ID)` where `COLLECTION_NAME` is used today. Then run
   them once per tenant, each against that tenant's PDFs under a per-tenant
   `data/manuals/{tenant_id}/`.
2. **A tenant admin surface** — create a tenant, upload its manuals, trigger ingestion.
   Out of scope here; the `tenants.csv` registry is the placeholder.
3. **Auth that carries the tenant** — the PIN form is a demo. Production is SSO / badge
   from the plant's identity system; the token must carry `tenant_id` and the backend
   must trust only that, never a value from the request body.
4. **Physical vs logical separation.** Shared SQLite with a `tenant_id` column is fine
   for a handful of tenants; a regulated deployment may need a database per tenant.
   Same call for the vector store: separate collections in one Chroma instance vs.
   one instance per tenant.

## Threat model

- **Cross-tenant retrieval leakage** — the only defence is that each request resolves to
  its own collection. `resolve_collection` must be the *single* place a collection name
  is chosen; a hard-coded `COLLECTION_NAME` anywhere in a request path is a bug.
  `tests/integration/test_tenant_isolation.py` pins this.
- **Cache poisoning across tenants** — mitigated: `tenant_id` is in the cache key and a
  lookup filters on it.
- **Tenant spoofing** — the backend must derive `tenant_id` from the authenticated
  principal, never from user input.
- **LangSmith traces** — today all traces go to one project (`dificuldades_e_
  oportunidades.md` #17). Per-tenant projects, or at least a `tenant_id` tag on every
  run, are needed before traces contain more than one customer's data.

## Migration from single-tenant

The current store *is* the `default` tenant's store — `resolve_collection("default")`
returns its exact name, so no data moves. Adding tenant `acme` only adds a new
collection; `default` is untouched.
