# Backend architecture

**Status:** design, plus a minimal proof (`api/main.py`, `Dockerfile`). The Streamlit
app is still the product; this documents the "Product-level" decoupled architecture and
what the proof does / doesn't cover.

## Why this is cheap to do

Phase 2 pulled every bit of business logic into `factory_floor/services.py` as plain
functions. So an API is a thin translation layer — read the request body into a
`DiagnosticRequest`, call `services.run_diagnostic`, shape the `DiagnosticResult` into
JSON. No logic is duplicated between the Streamlit app and the API; both are clients of
`services`.

## The proof (`api/main.py`)

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | liveness + the configured tenant |
| `POST` | `/diagnose` | one blocking diagnostic turn via `services.run_diagnostic` |

The vector store and the model are FastAPI dependencies (`get_vectorstore`, `get_llm`)
so tests override them with a fake-embedding store and a fake model — `tests/integration/test_api.py`.

Run locally: `uvicorn api.main:app --reload`
Container: `docker build -t factory-floor-api . && docker run -p 8000:8000 -e OPENAI_API_KEY=… -v "$PWD/data:/app/data" factory-floor-api`

## The full intended surface (not built)

| Method | Path | Notes |
|---|---|---|
| `POST` | `/diagnose` | blocking (built) |
| `POST` | `/diagnose/stream` | Server-Sent Events; the phase-4 "buffer the stream, gate, then emit" rule applies — with the gate active the SSE stream is a spinner event then the final answer event |
| `POST` | `/resolutions` | `machines.append_resolution_event` — operator records what they did |
| `POST` | `/resolutions/{id}/export` | `audit.export_to_cmms` |
| `GET` | `/audit` | `audit.get_audit_trail`, filtered by the caller's tenant |
| `POST` | `/auth/login` | `identity.authenticate` -> a bearer token carrying `operator_id` + `tenant_id` |

Every handler stays a thin wrapper over `factory_floor/*`.

## Deployment shape

```
            ┌─────────────┐
  clients ──▶ load balancer│  (sticky sessions only for the SSE endpoint)
            └──────┬──────┘
          ┌────────┴────────┐
     ┌────▼────┐       ┌────▼────┐     stateless API replicas
     │ uvicorn │  ...  │ uvicorn │     (api.main:app, one container each)
     └────┬────┘       └────┬────┘
          └───────┬─────────┘
        ┌─────────┼───────────┬───────────────┐
   ┌────▼────┐ ┌──▼───────┐ ┌─▼────────────┐ ┌▼─────────────┐
   │ Chroma  │ │ audit DB │ │ answer cache │ │ secrets vault│
   │ (per-   │ │ (Postgres│ │ (Redis, or   │ │              │
   │  tenant │ │  in prod,│ │  Chroma)     │ │              │
   │  coll.) │ │  not     │ │              │ │              │
   │         │ │  SQLite) │ │              │ │              │
   └─────────┘ └──────────┘ └──────────────┘ └──────────────┘
```

- **Stateless replicas.** `st.session_state` (conversation history, session cost) moves
  to the client or a session store; the API takes `chat_history` in the request.
- **SQLite -> Postgres.** WAL SQLite is fine for the single-node app; multiple API
  replicas writing concurrently need a real server DB. The `audit.py` API
  (`record_recommendation`, `get_audit_trail`, `DailyLedger`) is the seam — swap the
  connection layer, keep the callers.
- **Answer cache -> Redis** for a shared cache across replicas (or accept a per-replica
  Chroma cache).
- **Secrets** come from the vault at deploy time (`docs/secrets.md`).
- **Streamlit** becomes a pure API client, or is replaced by a SPA.

## Not in scope for the proof

Auth, rate limiting per principal, the streaming endpoint, the resolution/audit
endpoints, Postgres, Redis, the compose file, CORS, and observability wiring. Each is a
thin addition on the same `services` / `audit` / `identity` seams.
