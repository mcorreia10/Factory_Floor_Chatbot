# The Factory Floor — working notes for Claude

This file exists so that opening Claude Code in this project folder on **any machine**
picks up the context and decisions from prior work sessions, not just the code state.
Read this before making changes. Keep it updated as the project evolves — append,
don't just overwrite, unless something below is confirmed stale.

## What this project is

Industrial Maintenance Copilot (RAG) for Siemens electric motors + VFDs. Full
architecture, corpus, and setup steps are in `README.md` — read that first for the
"what". This file is for the "why" and the non-obvious gotchas.

Current milestone status (confirmed stale as of 2026-08-24, updated): **all 9 roadmap
items and all 7 bootcamp core requirements are done** — RAG, agent, tracing, vision,
evaluation, deployment, and the Safety Validator (the last as an audit utility, not a
live blocking gate — see the 2026-08-21 note near the end of this file). What's left is
explicitly non-core: an SDS corpus, hybrid/lexical retrieval for exact fault codes, and
the 3-page project document — see `dificuldades_e_oportunidades.md`. The history below
(2026-08-18 through 2026-08-19) is kept for context on how earlier milestones were
built; don't re-derive it from scratch.

Note (2026-09-03): the dated session notes that used to live in `README.md` moved to
`docs/project_log.md`, verbatim. Older notes in this file that cite "README's
2026-08-25 note" (and similar) now mean that file — the README is the front door.

## 2026-08-27 — Professionalization work started (branch `professionalization`)

A multi-phase "production-readiness" effort is underway on a dedicated branch
`professionalization`, to be merged back to `master` with a single `--no-ff` merge only
when the whole thing is green. The plan lives at
`~/.claude/plans/como-podemos-planear-ent-o-effervescent-cocoa.md`. Phases: 0 test
harness + CI · 1 Settings object + secrets seam · 2 service layer (`services.py`) ·
3 cost control · 4 blocking safety gate · 5 audit trail + operator identity + writable
history · 6 semantic cache · 7 multi-tenant (design-only + seams) · 8 minimal FastAPI +
Dockerfile. Every new behaviour is **opt-in / off by default** — a fresh clone with only
`OPENAI_API_KEY` must behave exactly as before. `.env` stays untouched (existing owner
decision); new knobs are `FACTORY_FLOOR_*` env vars, documented in `.env.example` only.

### Testing (phase 0 — done)

Automated tests now exist alongside the manual convention below (they do not replace the
`nbconvert` sweep + live Playwright smoke, which still run at each phase boundary).

- **Env:** the project runs on the conda `base` env (`/opt/miniconda3/bin/python`,
  Python 3.13) — *not* the Framework `python3` 3.14 on PATH, which only has a partial
  install. Run tests with `/opt/miniconda3/bin/python -m pytest` or
  `make test PYTHON=/opt/miniconda3/bin/python`.
- **Layout:** `tests/unit/` (fast, no network — the default) and `tests/integration/`
  (build a tiny real Chroma store with `DeterministicFakeEmbedding`, no API key).
  Markers `unit` / `integration` are auto-applied by directory (see `tests/conftest.py`).
  `llm`-marked tests need a real key and are skipped in CI / by `make test`.
- **Fake models:** every `factory_floor` entry point takes `llm=` — `conftest.py` has
  `make_fake_llm` (a `GenericFakeChatModel`) and `make_structured_llm` (supports
  `.with_structured_output()`, which `GenericFakeChatModel` does not).
- **`langchain_openai` imports slowly here (~6–19 s)** because it transitively pulls in
  `transformers` + `torch`. The `_no_network` autouse guard therefore does *not* import
  anything — it patches only already-imported modules — so a pure-logic test file stays
  fast. First test that touches `factory_floor.rag`/`.agent` pays the import cost once.
- **Config:** `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.ruff]` — minimal
  `select = ["E4","E7","E9","F"]`, source only, notebooks excluded). `requirements-dev.txt`.
  CI: `.github/workflows/ci.yml` (ruff + `pytest -m "not llm"` on py3.12/3.13, no key).

### Settings + secrets (phase 1 — done)

- `factory_floor/config.py` gained a frozen `Settings` dataclass + `get_settings()`
  (`@lru_cache`). Every field defaults to today's behaviour; `FACTORY_FLOOR_*` env vars
  override. `.env` untouched; new knobs documented in `.env.example` only.
- **Cache trap:** `get_settings()` is lru_cached and is first called during
  `configure_tracing()` at package import — which in `app.py`'s flow is *before*
  `load_dotenv()`. `app.py` therefore calls `get_settings.cache_clear()` right after
  `load_dotenv()`. Notebooks load `.env` first, so they're fine. Tests clear it via an
  autouse fixture.
- `factory_floor/secrets.py::get_secret()` — default `env` backend == `os.getenv`
  (current behaviour). `aws`/`vault`/`doppler`/`sops` are `NotImplementedError` seams;
  `docs/secrets.md` has the deploy-time flow.

### Service layer (phase 2 — done)

- `factory_floor/services.py` — a diagnostic turn as plain functions, zero Streamlit:
  `DiagnosticRequest`/`DiagnosticResult` dataclasses, `check_typo`, `classify_photo`,
  `build_diagnostic_retriever`, `run_diagnostic` (blocking + streaming), `assemble_turn`.
- `app.py::submit_turn` is now: read widgets/session -> build `DiagnosticRequest` ->
  `services.run_diagnostic(..., stream=True)` -> `st.write_stream` -> `services.assemble_turn`.
  The 3 `@st.cache_*` resource loaders stay in `app.py`.
- **`run_diagnostic` is the hook point for phases 3-6** (cost, safety gate, audit, cache).
  In phase 2 it is a thin pass-through to `factory_floor.agent`.
- **Streaming contract:** `run_diagnostic(stream=True)` returns `(generator, result)`;
  `result.answer`/`documents`/`sources`/`tool_trace` stay empty until the caller fully
  consumes the generator (the wrapper copies them from the `StreamedAgentRun` at the end).
- **Test gotcha:** `create_agent` calls `model.bind_tools()` on every step, which
  `GenericFakeChatModel` doesn't implement. `conftest.py::make_agent_fake_llm` subclasses
  it with `bind_tools -> self` so integration tests drive the real `create_agent` graph
  (no tool calls) without an API key.
- **Not done here (still deferred):** the `use_container_width=True` -> `width="stretch"`
  deprecation (now ~9 call sites in `app.py`) — cosmetic, needs a live render to verify.
- **Full nbconvert sweep + Playwright are the phase-boundary check** (real API cost) —
  run before pushing / merging, not per commit.

### Cost control (phase 3 — done)

- `factory_floor/cost.py`: `count_tokens` (tiktoken, `o200k_base` fallback),
  `MODEL_PRICING` (USD/1M tokens, dated 2026-08), `estimate_cost`, `UsageAccumulator`
  (per-session, `as_dict`/`from_dict`/`merge`), `CostTrackingCallback`
  (`BaseCallbackHandler`), `DailyLedger` (append-only JSONL, folds into SQLite in
  phase 5), `check_spend_cap` / `SpendCapExceeded`.
- `services.run_diagnostic` now: pre-flight `check_spend_cap` against the day's ledger
  total + a conservative `PENDING_TURN_ESTIMATE_USD` (0.05); on `SpendCapExceeded`
  returns a `blocked=True` result (streaming: `(empty_generator, result)`). Otherwise
  attaches `CostTrackingCallback` via `config={"callbacks": [cb]}` on the agent call
  (propagates to the reranker sub-call too), then writes a ledger row and sets
  `result.cost`. Streaming finalises cost only after the generator is consumed.
- `rag.get_llm()` now sets `stream_usage=True` on `ChatOpenAI` so streamed agent runs
  report `usage_metadata`. **VERIFY in the Playwright/nbconvert boundary check:** the
  usage-bearing final chunk has empty `content`, so `stream_diagnostic_agent`'s
  `message_chunk.content` filter should drop it — confirm no stray usage/JSON text
  streams to the operator and notebook 07/12 output is unchanged. If it misbehaves,
  drop `stream_usage=True` and rely on the callback's tiktoken fallback for streamed
  runs (input under-counted).
- `app.py`: sidebar shows "Session LLM cost: $x · N calls" and, when a cap is set, a
  today-vs-cap progress bar; a blocked turn shows `st.error(result.message)` and is not
  appended. Session total lives in `st.session_state["usage"]` (merged each turn).
- Off by default: no `FACTORY_FLOOR_DAILY_SPEND_CAP_USD` -> cap check is a no-op; the
  only effect is `result.cost` / the sidebar line being populated. (Ledger storage moved
  to the audit SQLite `cost_events` table in phase 5 — see below.)

### Blocking safety gate (phase 4 — done)

- `safety.py` gained (additive; the 3 audit functions for notebooks 09/10 are untouched):
  `enforce_safety(answer, *, llm, mode, language)`, `SafetyGateResult` (`action` =
  `pass` | `rewritten` | `held`), `SAFETY_REWRITE_SYSTEM_PROMPT`, `FIXED_HELD_FALLBACK`,
  `_tokens_preserved` (a rewrite must keep every `[SOURCE n]` and fault code).
- Logic: keyword check first; **no physical action -> pass, no LLM judge call** (cheap
  path for clarifying questions / explanations). Otherwise run the judge; if it fails,
  `mode="rewrite"` does one rewrite + re-check + citation-survival check (deliver if it
  now passes and kept citations, else hold); `mode="block"` holds without rewriting.
  Held answers are replaced with `FIXED_HELD_FALLBACK`.
- `services.run_diagnostic` runs the gate on the finished answer; `result.answer` becomes
  the gate's `delivered_answer`, `result.safety` = `SafetyGateResult.as_dict()`.
  **Streaming:** with the gate active the raw tokens can change, so the stream is drained
  silently and the gated text is emitted as one chunk (spinner-then-answer). With
  `mode="off"` live token streaming is preserved. `safety_gate_on_stream` exists in
  Settings but currently both values behave the same (Streamlit can't un-write a stream).
- `app.py`: `held` -> `st.error` + the fallback text is what's shown; `rewritten` ->
  `st.info`; `render_turn` shows a small badge. `turn["safety"]` carries the verdict.
- Default `mode="rewrite"` — so the default experience now sometimes rewrites/holds an
  answer. Set `FACTORY_FLOOR_SAFETY_GATE_MODE=off` for the exact pre-phase-4 behaviour.
- **Test note:** `enforce_safety` needs a model that both judges (`with_structured_output`)
  and rewrites (`invoke`) — `conftest.make_gate_llm`. `run_diagnostic`'s gate wiring is
  integration-tested with `enforce_safety` monkeypatched (the agent fake can't do
  structured output). Live behaviour: `tests/integration/test_safety_gate_live.py`
  (`-m llm`).

### Audit trail + operator identity + writable history (phase 5 — done)

- `factory_floor/audit.py` — stdlib `sqlite3`, `PRAGMA journal_mode=WAL` (the answer to
  the concurrent-write race in `dificuldades_e_oportunidades.md` #6). Tables:
  `recommendations`, `recommendation_sources`, `tool_calls`, `cost_events`,
  `resolution_events`. `record_recommendation(result, req)`, `get_audit_trail(...)`,
  `append_resolution_event(...)`, `get_resolution_events(...)`, `export_to_cmms(...)`.
- **The phase-3 JSONL cost ledger is gone** — `DailyLedger` now lives in `audit.py`
  (backed by `cost_events`), re-exported from `cost.py` for existing imports.
  `Settings.cost_ledger_path` / `FACTORY_FLOOR_COST_LEDGER_PATH` removed. When the audit
  trail is enabled the cost row is written by `record_recommendation` (linked to the
  recommendation); disabled -> `DailyLedger.record` writes a standalone row. The spend
  cap reads `DailyLedger().today_total()` either way.
- `factory_floor/identity.py` — `Operator`, `authenticate(operator_id, pin)`,
  `list_operators()`, `hash_pin` (PBKDF2-SHA256, 100k rounds). Backed by committed
  `operators.csv` (root, like `machines.csv`): 3 fake operators, **PINs 1234 / 5678 /
  4321** — printed on purpose so the demo and tests can sign in; a real deployment gets
  these rows from the plant's identity system.
- `machines.py`: `get_machine_history(machine_id, include_resolutions=False)` — default
  False keeps notebook 05 and the agent's history tool unchanged; True unions live
  `resolution_events` (tagged `event_type="operator_resolution"`). The static CSV stays
  read-only. New `machines.append_resolution_event(...)` delegates to `audit`.
- `services.run_diagnostic._finalize`: if `settings.audit_enabled` (**default true**)
  call `audit.record_recommendation` and set `result.audit_id`; the turn dict carries
  `audit_id`.
- `app.py`: optional sign-in gate (`FACTORY_FLOOR_REQUIRE_LOGIN`, default off); operator
  chip + sign-out in the sidebar; a "Record what you actually did" expander per machine
  with a "Save to machine history" + "Send to CMMS/ERP (demo)" pair; the history
  expander now shows static + live rows.
- **Deviation from plan:** the resolution-recording UI is one expander after the turns
  (not per-turn) — Streamlit rerun/key management makes per-turn text areas + buttons
  fiddly. It ties to the last turn's `audit_id`.
- Paths: `FACTORY_FLOOR_AUDIT_DB_PATH` (default `data/audit.sqlite3`),
  `FACTORY_FLOOR_CMMS_OUTBOX_PATH` (default `data/cmms_outbox.jsonl`) — both gitignored,
  and pointed at tmp_path by `conftest._isolate_runtime_state` so tests never touch the
  real files.

### Semantic answer cache (phase 6 — done)

- `factory_floor/cache.py::SemanticCache` — its own Chroma collection
  `factory_floor_qa_cache` in `data/qa_cache/` (`hnsw:space=cosine`, so
  `similarity = 1 - similarity_search_with_score` distance is a real 0-1 number — the
  l2 relevance-fn trap from 2026-08-25 is avoided). Opt-in via
  `FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED`.
- Scoped by machine / equipment_type / language / tenant. **Fault-code questions**
  (`_code_signature` non-empty — catches `F3OO21` typos too via `extract_possible_codes`)
  require an exact normalized-question match via a metadata `.get(where=...)` — no
  embedding call, never a near-miss. Prose questions: cosine similarity >= threshold
  (default 0.95). A code-free question is never served from a code-bearing entry.
- `store()` only takes first-turn, no-photo questions whose safety action is `pass` or
  `rewritten` (never `held`, never blocked). `version_stamp` (embedding model + main
  collection name) + a TTL (720h) bound staleness; a manual-store rebuild should also
  call `SemanticCache().clear()` (documented, not automatic).
- `services.run_diagnostic`: cache lookup runs **before the spend-cap check** (a hit
  costs nothing). A hit builds a `DiagnosticResult` with `cache_hit=True`, zero cost,
  `safety={"action":"pass","reason":"served from semantic cache"}`, still writes an
  audit row (flagged `cache_hit=1`), skips the agent + gate. Streaming: yields the
  answer as one chunk. Miss -> normal path -> `store()` in `_finalize`. The cache reuses
  the **main store's embedding function** (`vectorstore.embeddings`) so a fake-embedding
  test store keeps the cache offline.
- `app.py`: sidebar shows the cache size + a "Clear answer cache" button when enabled;
  `render_turn` shows a "⚡ Answered from the cache" caption; `turn["cache_hit"]` persists it.
- `FACTORY_FLOOR_SEMANTIC_CACHE_DIR` (default `data/qa_cache/`, gitignored) — pointed at
  tmp_path by the conftest isolation fixture.

### Multi-tenant (phase 7 — design-only + seams)

- `factory_floor/tenancy.py`: `resolve_collection(tenant_id)` (`default`/None ->
  `settings.collection_name`; else `"{base}__{sanitised}"`), `list_tenants()` /
  `get_tenant()` off `tenants.csv` (root, committed, one `default` row).
- `services.load_tenant_vectorstore(tenant_id)` uses it; `app.py::load_rag_components`
  is now keyed on `tenant_id` (its `@st.cache_resource` key), read from
  `operator["tenant_id"]` (default `"default"`). For `default` the resolved collection
  name is byte-identical to before — nothing changes for the running app.
- Most tenant seams were already threaded by earlier phases: `tenant_id` on
  `DiagnosticRequest`, on the `recommendations`/`cost_events`/`resolution_events` tables,
  in `SemanticCache`'s key, and on `operators.csv` rows; `DailyLedger.today_total` and
  `get_audit_trail` filter on it.
- **Deviation:** notebooks 01/02 were NOT edited to read a `TENANT_ID` env var
  (re-running the ingestion notebooks risks the demo-critical `data/vectorstore/`). The
  exact one-line change they need is written in `docs/multi_tenancy.md`.
- The design + threat model is `docs/multi_tenancy.md`. Not built: tenant admin surface,
  per-tenant ingestion pipeline, real auth carrying the tenant, physical DB separation.
- Tests: `tests/unit/test_tenancy.py`, `tests/integration/test_tenant_isolation.py`
  (two collections in one persist dir; retrieval scoped to one never returns the other's
  docs).

### FastAPI proof + Dockerfile (phase 8 — minimal)

- `api/main.py` — `GET /health`, `POST /diagnose` (blocking, via `services.run_diagnostic`).
  The vector store and model are FastAPI `Depends` (`get_vectorstore`, `get_llm`) so
  `tests/integration/test_api.py` overrides them with a fake-embedding store + fake model.
  `load_dotenv()` at the top for local dev (no-op in a real deployment where the runtime
  injects the key).
- `Dockerfile` — python:3.13-slim, `uvicorn api.main:app`. `data/` (manuals + vector
  store) is **not** baked in — mount it at runtime. **Not built** in this session
  (docker not available on the box); the image is the design artifact.
- `requirements.txt` gained `fastapi` + `uvicorn[standard]` (clearly marked optional —
  not needed for the Streamlit app or notebooks); `requirements-dev.txt` gained `httpx`.
- `docs/backend_architecture.md` — the full intended surface (SSE `/diagnose/stream`,
  `/resolutions`, `/audit`, `/auth/login`), the LB / stateless-replica / SQLite→Postgres
  / Redis-cache deployment shape, and what the proof deliberately omits.
- Verified live: `uvicorn api.main:app` boots; `GET /health` -> `{"status":"ok",...}`;
  a real `POST /diagnose` returned a grounded answer + 5 sources + tool_trace + run_id +
  cost_usd + audit_id, 200 OK ($0.0016).
- **Note:** `api.main` imports `factory_floor` before `load_dotenv()`, so it does the
  same `get_settings.cache_clear()` dance as `app.py`. uvicorn cold start is ~20s here
  because `langchain_openai` pulls in transformers+torch.

## 2026-08-25 — Exact fault-code lookup (difficulty #1 closed)

Non-obvious things worth keeping; the full write-up with numbers is in
`dificuldades_e_oportunidades.md` #1 and README's 2026-08-25 note.

- **Chroma already does exact substring search — nobody had noticed.** `chromadb 1.5.9` +
  `langchain-chroma 1.1.0` accept `where_document={"$contains": "F30021"}`, it combines with the
  existing metadata `filter`, and `as_retriever(search_kwargs=...)` forwards it verbatim. The
  local FTS index is `tokenize='trigram'`, i.e. real substring matching with no tokenisation
  trap. So the whole fix needed **no new dependency, no BM25, no re-indexing** — the long-planned
  "hybrid retrieval" turned out to be one already-present feature. Check what the installed
  stack does before reaching for a library.
- **`vectorstore._collection.get(where_document=..., where=...)` is a pure lexical lookup** with
  no embedding call at all. That is what `fault_codes.lookup_code()` uses — embedding a query
  whose whole point is exactness would be both wasted work and self-defeating.
- **Literal matching alone is not enough**, and this is the trap the 2026-08-18-d audit already
  hit once: most occurrences of a code are cross-references from other pages. `F30021` matches 8
  chunks, and the *first* is "See also: F30021 Note: ..." on p.62; the definition is on p.908.
  `F07011` matches **53** chunks. `score_occurrence()` reproduces the old best-match-per-code
  rule (penalise `See also:`/`Note:` before, and code-lists after; reward a `Power unit:`/`Drive:`
  prefix). Verified: picks the definition 8/8, and drives the definition-page metric from 18% to
  100%.
- **Exact hits must bypass the reranker.** They are pinned ahead of the semantic results and never
  passed to `rerank_documents()` — an LLM reordering them can bury the one chunk that defines the
  code, which is the entire point of the lookup.
- **When a code is unknown AND the query has nothing else, suppress the semantic results
  entirely** (`CodeAwareRetriever._has_searchable_text`). Otherwise the UI shows four real manual
  pages as "Sources retrieved" underneath an answer that correctly says the code is undocumented
  — the exact misleading pairing this change exists to remove. If the operator *did* describe a
  symptom too, the semantic results still come through.
- **`difflib` is the wrong tool for these typos.** `F3OO21` vs `F30021` scores below any usable
  cutoff (measured: no match even at 0.75) because O and 0 are unrelated characters to it. What
  works is normalising the actual display confusions (O↔0, I↔1, S↔5, B↔8, Q, L, Z) and then
  testing exact membership. The loose pattern used to *find* a malformed code requires ≥3 real
  digits, otherwise ordinary words like `FOSSIL` and `BOBBIN` match `[FA][0-9OQILSBZ]{5}`.
- **A typo is not an unknown code.** Suggestion is surfaced to the operator as a question with two
  buttons, never auto-applied and never fed to the LLM — answering about a code the operator
  never asked about is exactly the failure being prevented. Owner-confirmed decision: for a
  genuinely unknown code, refuse cleanly and do **not** list similar codes.
- `fault_codes.csv` holds **368** codes (206 F / 162 A) — more than the 335 of the 2026-08-18-d
  audit, because that one scanned only the two List Manuals while `build_fault_code_index.py`
  scans every chunk in the store.

## 2026-08-24 — Git repository confirmed real (was wrongly listed as an open gap)

A fresh analysis pass found this repo already has a real `origin` remote
(`https://github.com/mcorreia10/Factory_Floor_Chatbot`) with 3 real commits — the
"real Git repository" gap noted on 2026-08-21 (README, this file,
`dificuldades_e_oportunidades.md`) was stale documentation, not an actual missing
deliverable. `git status` was clean apart from a benign `.claude/settings.local.json`
permission-allowlist diff. Updated README.md's compliance-checklist note and this
file's status line to drop it from the "what's left" list. **Note:** `dificuldades_e_
oportunidades.md` never actually named this item as one of its own numbered
difficulties (it was only cross-referenced from README/CLAUDE.md), so nothing needed
to change there.

## Decisions made deliberately (don't "fix" these without asking)

- **`.env` is kept as-is on purpose.** It contains API keys for services this project
  doesn't use (Pinecone, HuggingFace, Tavily, Anthropic, Google, LangSmith) alongside
  the `OPENAI_API_KEY` this project actually needs — it looks like a shared/personal
  `.env` reused across projects. The owner was told about this explicitly and chose not
  to split it out ("não separes o .env, sei o que faço"). Don't split or "clean" it
  without being asked again.
- **`download_manuals.py` was never the actual path used to get the 11 PDFs** (they
  were fetched manually). It was live-tested this session (one real HTTP GET per
  distinct mirror host in `manual_sources.csv`, into a scratch dir, not overwriting
  `data/manuals/`) and all 4 hosts returned valid PDFs — so it's a genuinely working
  reproducibility path if `data/manuals/` is ever lost, kept as documented Step 1.
- **`manual_sources.csv` has an `equipment_type` column** (VFD / electric_motor) that
  `factory_floor/ingestion.py` reads directly. This replaced the earlier
  filename-substring-guessing approach, which was fragile for future manuals.

## Streamlit gotchas hit this session (real bugs, now fixed — don't reintroduce)

1. **Any `st.button`/`st.download_button` click triggers a full script rerun.** Nothing
   that needs to survive a rerun can live in a plain local variable — it has to be in
   `st.session_state` (see `st.session_state["turns"]` in `app.py`). This is why the
   "download this page" buttons don't wipe the conversation.
2. **`st.columns()` anchors content to where it was *called*, not to script order.**
   Re-entering `with some_col:` later in the script does **not** move that content
   further down the page — it still renders in the original column's fixed position.
   The follow-up question box was originally misplaced at the top of the page for this
   reason; fixed by creating a *new* `st.columns()` call after the turn-rendering loop
   instead of reusing `form_col`.
3. **Deleting a widget-bound `st.session_state[key]` does not reliably clear that
   widget visually** (the frontend can keep the old value since the widget id/key
   didn't change). The reliable fix is to change the widget's `key` after each
   submission (see `followup_key` counter in `app.py`), not `del` + rerun.

## Feature-specific notes

- **Conversation history** (`factory_floor/rag.py`): follow-up questions are rewritten
  into standalone questions via `contextualize_question()` *before* retrieval — this
  was a deliberate choice (Option A: LLM reformulation) over cheaper context-stuffing,
  because elliptical follow-ups retrieve poorly against the vector store otherwise.
  Costs one extra LLM call, but only on turns after the first (`chat_history` empty →
  skipped, so turn 1 is byte-for-byte identical to pre-history behavior).
- **Page-level PDF download**: `factory_floor/manuals.py::extract_page_pdf()` extracts
  a single page via `pypdf`. Wrapped in `@st.cache_data` in `app.py`
  (`cached_page_pdf`) because it re-runs on every rerun for every visible source across
  every turn — without caching this got noticeably slow once a conversation had 2+
  turns with evidence expanded.
- **Language selector**: `ask(..., language=...)` only affects the *next* answer
  generated — past turns keep whatever language they were originally answered in
  (not retroactively translated). The system prompt has an explicit rule to never
  translate fault codes / parameter numbers / equipment names, since operators need to
  recognize them on the physical equipment display.

## Testing convention established this session

Don't just run notebooks and call it verified — for any `app.py` UI change, actually
launch Streamlit and drive it with Playwright (headless Chromium via a small Node
script, not `chromium-cli` — wasn't available in this Windows environment, so a plain
Playwright script under `node` was used instead) before declaring it done. Several real
bugs (the two Streamlit gotchas above) were only caught this way, not by reading the
code. Notebooks 01–04 should also be re-executed (`python -m nbconvert --to notebook
--execute --inplace <path>`) after any `factory_floor/` change to confirm no
regression, since `jupyter nbconvert` isn't on PATH in this environment — use
`python -m nbconvert` instead.

## Known non-blocking issues (not yet fixed)

- `app.py` uses `use_container_width=True` in 4 places (`st.dataframe` and 3
  `st.button` calls) — deprecated by Streamlit, to be replaced by `width="stretch"`
  after 2025-12-31. Flagged to the owner, not yet actioned.
- Retrieval always returns top-5 chunks with no relevance threshold — off-topic
  questions (tested: "what do you think about football") still populate the "Sources
  retrieved" table even though the model correctly refuses to answer from them. Could
  add a similarity cutoff later if this confuses operators.

## Next steps discussed (not started)

Roadmap items from the README, in the order they were prioritized in conversation:
vision (extracting diagrams/nameplates from PDFs — currently `PyPDFLoader` only pulls
text, losing wiring diagrams), agent/routing (auto-select equipment_type filter,
decide when retrieval is needed — **note:** the equipment_type auto-filter itself is
now done, see below; what's left here is the *decide when retrieval is needed* part,
i.e. a real orchestrator), persistent memory (cross-session, not just in-conversation
history), safety/eval validation (systematic groundedness/hallucination testing —
currently only enforced by prompt wording, never measured).

## 2026-08-18 — Maintenance history feature (README point 10)

- **New per-machine/asset concept, introduced from scratch.** Before this, the app had
  no notion of a specific physical unit — only `equipment_type` (a manual *category*:
  VFD vs electric_motor). Added a fixed registry of 20 machines (10 electric motors +
  10 VFDs) in `machines.csv`, plus a simulated fault/repair log in
  `maintenance_history.csv` (2-6 events per machine, 87 total), both at repo root —
  **deliberately not under `data/`** — mirroring `manual_sources.csv`'s precedent:
  small, structured, hand-generated-then-committed reference tables live at root;
  `data/` is for bulk generated artifacts (PDFs, the Chroma index).
- **Both CSVs were generated once with a fixed random seed and committed as static
  files.** `factory_floor/machines.py` only *reads* them (`load_machines()`,
  `load_maintenance_history()`, `get_machine_history(machine_id)`) — no
  generation/randomization logic lives in the package, same spirit as
  `manual_sources.csv` never being regenerated at runtime.
- **VFD fault events carry a real-looking Siemens SINAMICS fault code** (F0001,
  F0002, F0003, F0011, F0021, F0022, F0035, F0051), deliberately chosen to match the
  fault-code system actually documented in the ingested List Manual PDFs, so the demo
  can connect a machine's own history to what the manuals say about that code.
  Electric motor events have no such fault-code system in their manuals, so they use a
  free-text `description` instead (bearing noise, insulation resistance, overheating,
  vibration) and leave `fault_code` empty — this distinction is deliberate, not a gap.
- **Decision: selecting a machine in the sidebar auto-filters the RAG search** by that
  machine's `equipment_type`, via `build_retriever(vectorstore, k=TOP_K,
  equipment_type=...)` — a parameter that already existed in `factory_floor/rag.py` but
  that `app.py` never used before this. Confirmed with the project owner before
  implementing (not a silent behavior change) — picking a VFD now restricts sources to
  VFD manuals, picking a motor restricts to electric_motor manuals.
- **Scope boundary, deliberate:** the maintenance history is *not* wired into the LLM's
  prompt/reasoning at this milestone — `ask()` in `factory_floor/rag.py` is untouched.
  The Streamlit UI only retrieves and *displays* the selected machine's history. Feeding
  it into the model's reasoning alongside the RAG Tool is the future Orchestrator
  Agent's job (a separate, later roadmap step), not this one.
- **UI placement, and a real layout lesson from this session:** the machine selector
  lives in `st.sidebar` (compact — just the picker + a one-line caption), specifically
  to sidestep gotcha #2 below (`st.columns()` anchoring). The maintenance-history table
  was *first* built inside the sidebar too, but a 9-column `st.dataframe` in the narrow
  sidebar truncated almost everything past `event_date` — confirmed visually via live
  browser testing, not caught by reading the code. Moved it to a full-width
  `st.expander` in the main area instead (placed before the two-column layout begins,
  so it's not nested inside any existing `with col:` block and isn't subject to gotcha
  #2 either). Lesson: a table with more than ~4-5 columns does not belong in the
  sidebar in this app, regardless of how convenient that placement seems structurally.
- **New session-state key: `st.session_state["selected_machine"]`**, assigned on every
  script run from the sidebar `st.selectbox`'s current value (mirrors how `turns` is
  handled) — read back inside `submit_question()` for the retrieval filter, and used to
  title the history expander. Verified live (browser-driven, per the testing convention
  below) that it survives: a failed empty-question submit, a successful multi-turn
  follow-up, and "← Back to start" (which clears `turns` but must not clear the machine
  choice — confirmed it doesn't).
- New `notebooks/05_maintenance_history.ipynb` — no LLM/vectorstore calls, so it runs
  without `OPENAI_API_KEY`; ran clean via `python -m nbconvert --to notebook --execute
  --inplace`.

## 2026-08-18-b — Maintenance history: owner feedback, not done yet

- **Owner explicitly said this milestone is not complete** ("não considero que este
  tema esteja ainda completo"). Reverted the PPTX roadmap slide's item 04 from "Done"
  back to **"On Going"** (progress stat back to 44% / 4 of 9), and reverted README's
  point 10 from ✅ to 🔧 in-progress. Don't re-mark either as done without asking again
  — the working code isn't in question, the *scope* is considered unfinished by the
  owner (most likely: no LLM/prompt integration yet, that's still deferred to the
  Orchestrator Agent step — see scope boundary note above).
- **Moved the "Maintenance history" expander** from its own block to *inside*
  `render_turn()`, immediately after the "Show retrieved evidence" expander and before
  `st.divider()` — explicit owner request, so the history reads as evidence attached to
  that specific answered turn rather than a static sidebar-adjacent panel. Hidden
  entirely when `machine_id == "GENERAL"` (see next point).
- **Added a "🌐 General question (search all manuals)" pseudo-machine** as the first
  entry in the sidebar selector (`GENERAL_MACHINE` dict in `app.py`, not in
  `machines.csv` — it's a UI-only sentinel, not a simulated asset). Its
  `equipment_type` is `""`, which `build_retriever()`'s existing `if equipment_type:`
  check already treats as "no filter" — no changes needed in `factory_floor/rag.py`.
  Verified live: a general question returned mixed VFD + electric_motor sources with
  no filtering, and correctly shows no maintenance-history expander at all.
- **Removed the "Example question" dropdown** from the question form entirely (and the
  `examples` list it read from) — owner wants operators to type their own question,
  not pick a canned one. The `placeholder=` text on the text_area was kept since it's
  just greyed-out illustrative text, not a selectable shortcut.
- New `test_questions.txt` at repo root (not part of the app/package, just a scratch
  file for manual testing) — 70 questions in English (manuals are English, so
  retrieval quality is best when the question is too): 40 "manual-style" questions (20
  motor / 20 VFD) plus 30 realistic shop-floor complaints (15 / 15) written like an
  operator would actually phrase them, referencing real `machine_id`s from
  `machines.csv` for testing alongside the sidebar selector.

## 2026-08-18-c — Fake fault codes bug (owner-caught, real fix)

- **The owner caught a real data-quality bug**: the VFD fault codes used in the first
  draft of `maintenance_history.csv` (F0001, F0002, F0003, F0011, F0021, F0022,
  F0035, F0051) were **invented** — a 4-digit format assumed from generic Siemens
  convention, never checked against the actual ingested PDFs. Asking the app about any
  of them (e.g. "F0051 code showed up after several hours of running") correctly
  triggered the system prompt's "documentation is insufficient" rule — not a crash, but
  a dead-end demo, because the code genuinely isn't in the corpus.
- **Root cause found by direct retrieval investigation** (`vs.similarity_search`
  against the real Chroma store, not just reading PDF text): the real Siemens SINAMICS
  G120/G120C fault list uses **5-digit codes** (`F3xxxx`, `F0xxxx`, `F07xxx` families),
  e.g. `F30001` Overcurrent, `F30002` DC link overvoltage, `F30003` DC link
  undervoltage, `F30021` Ground fault, `F30035` Air intake overtemperature, `F30037`
  Rectifier overtemperature, `F07807` Short-circuit/ground fault at motor terminals,
  `F30950` Internal software error — all confirmed present via exact-string match on
  retrieved chunks, with real cause/remedy text extracted alongside.
- **Second, more important finding: bare fault-code queries retrieve poorly even for
  real codes.** Tested systematically — querying `"Fault code {code}, what should be
  checked and what is the remedy?"` for ~18 real candidate codes at k=5 (and even
  k=15) hit the correct page for only a handful of them, because that generic
  boilerplate phrasing dominates the embedding and pulls in whatever fault-list page is
  semantically "closest to a generic fault question" rather than the target code's
  specific page — these list-manual pages are dense with many near-identical
  neighboring codes. **Pairing the code with its actual symptom/keyword** (e.g. "F30021
  ground fault" instead of bare "F30021") fixed retrieval reliably. This is a structural
  limitation of pure embedding-based retrieval for exact alphanumeric identifiers, not
  a bug to "fix" within this milestone — flagged as a real limitation, relevant to the
  future Evaluation roadmap step (09). Don't assume any fault code the model or a demo
  question mentions will retrieve well unless it's one of the 8 verified above, or has
  been checked the same way.
- **Fix applied**: regenerated `maintenance_history.csv` (same seed=7, same
  2-6-events-per-machine structure, motor-side data untouched — only the VFD fault
  vocabulary changed) using the 8 verified real codes with their real cause/remedy
  text as `description`/`action_taken`. Verified end-to-end afterward: `ask()` on
  "F30021 ground fault showed up after several hours of running, what should be
  checked?" now returns a real grounded 8-step answer citing the actual manual pages.
  Updated `README.md`'s example code list and `test_questions.txt`'s VFD questions to
  match (all now pair code + short symptom, per the finding above). Re-ran
  `notebooks/05_maintenance_history.ipynb` afterward so its printed output reflects the
  corrected data.

## 2026-08-18-d — Full fault/defect code audit (README's "Next step", now done)

- **Why a full audit, not just the 8 spot-checked codes**: the owner correctly pointed
  out that 8 codes checked via semantic search (previous note) wasn't a real audit —
  semantic search is exactly the retrieval method already shown to be unreliable for
  exact codes, so "verifying" codes that way was checking the method against itself.
  This time: **direct regex extraction over the raw PDF text** of
  `Siemens_SINAMICS_G120C_List_Manual.pdf` and `Siemens_G120_CU240BE2_List_Manual.pdf`
  via `pypdf`, pattern `[FA]\d{5}` — a full page-by-page scan, not a sample. Found 335
  unique codes total (191 `F` faults, 144 `A` alarms).
- **First extraction attempt had a bug worth remembering**: naively keeping the first
  regex match per code picked up incidental mentions (a code number referenced inside
  an unrelated note/cross-reference on an earlier page) instead of the code's real
  definition — e.g. `F30021` first matched a "Note: this parameter is only relevant
  for chassis power units" line, not its actual "Power unit: Ground fault" definition
  several pages later. Fixed by collecting *all* matches per code and scoring them:
  reject anything starting with `Note:`/`Notice:`/`Danger:`/`Description:`/`Warning:`,
  prefer text starting with `Power unit:`/`Drive:`/`CU:`/`PU:`/`TM:`/`SI `/`PROFIBUS:`
  etc. **Lesson for future extraction work on these manuals: a fault-code table has
  many incidental cross-references to other codes scattered through the document —
  always take the best-scoring match, never the first.**
- **Picked 17 of the 191 real F-codes** for the simulated data — not all 191, since
  that's far more than 87 history events need and many are narrow internal/software
  states (`F01xxx` boot/firmware errors, `SI P1/P2` safety-function internals) not
  relevant to a shop-floor maintenance log. Selected for category diversity instead:
  `F30001` overcurrent, `F30002`/`F30003` DC link over/undervoltage, `F30004`/`F30035`/
  `F30037` three overtemperature flavors (heat sink, air intake, rectifier), `F30011`
  line phase failure, `F30021` ground fault, `F30059` internal fan, `F30805` EEPROM
  checksum, `F30950` internal software error, `F07011` motor overtemperature, `F07016`
  motor temp sensor fault, `F07801` motor overcurrent, `F07807` short-circuit/ground
  fault at motor terminals, `F07860` external fault 1, `F07901` motor overspeed. All 17
  verified individually: `vs.similarity_search(code + keyword, k=5, filter=VFD)` hits
  the chunk containing that exact code string, and a full `ask()` call on two of them
  (`F07011`, `F07901`) returned correct, real, cause-grounded troubleshooting steps.
- **Electric motors: confirmed with the same regex, zero matches across all 5 motor
  manuals** (`Siemens_SIMOTICS_SD_Operating_Instructions.pdf`,
  `..._SD_1LE7_Operating_Instructions.pdf`, `..._GP_1LE1_Operating_Instructions.pdf`,
  `..._GP_SD_DP_Engineering_Manual.pdf`, `..._GP_SD_XP_DP_Catalog.pdf`). Motors
  genuinely have no numeric fault-code system — this was asserted from general
  reasoning in the previous session's note, now actually checked. Also extracted real
  terminology via keyword-occurrence counts + surrounding-context snippets: "abnormal
  noise"/"unusual noise" (used when describing a machine not running smoothly or when
  manually turning the rotor), "insulation resistance... below the specified value",
  "condensation" (transport/storage humidity spec), "re-greasing" tied to the
  lubricant plate interval, "direction of rotation" checks done with the motor
  uncoupled. Rewrote all 10 motor `description`/`action_taken` pairs in
  `maintenance_history.csv` to use this real phrasing instead of generic wording.
- **Regenerated `maintenance_history.csv`** (new seed=11, since the vocabulary itself
  changed — no reason to preserve the old seed's exact rows) — 77 events across 20
  machines, 17-code VFD vocabulary / 10-type motor vocabulary. Updated
  `test_questions.txt` (17 VFD manual-style Qs + 17 shop-floor VFD scenarios, one per
  verified code) and `README.md`'s code table to match. Re-ran
  `notebooks/05_maintenance_history.ipynb`.
## 2026-08-18-e — Maintenance history moved out of per-turn loop

- **Owner-caught UX bug**: the maintenance history expander (added in the
  2026-08-18-b note) lived inside `render_turn()`, so it repeated identically after
  *every* turn in a multi-turn conversation — same machine, same table, shown 3 times
  in a 3-turn conversation. Confirmed live via browser: 3 follow-ups on the same VFD
  produced 3 identical "Maintenance history — VFD-06" expanders.
- **Fix**: removed the expander from `render_turn()`; it's now rendered once, after
  the `for idx, turn in enumerate(...): render_turn(turn, idx)` loop, inside the same
  `if st.session_state["turns"]:` block that already guards the follow-up question
  box — so it sits once at the end of the conversation, right before "Ask a follow-up
  question", not attached to any individual turn. Verified live: a 2-turn conversation
  now shows exactly one history expander, positioned after the last turn's "Show
  retrieved evidence".
- **Also observed while testing this** (not fixed, just noted): right after toggling
  an expander, `st.button` labels ("Send follow-up", "← Back to start") sometimes
  paint as an empty colored bar for a moment — confirmed via `get_page_text` that the
  label text is present in the DOM the whole time, so it's a transient CSS/repaint
  timing issue, not a data-loss bug. Self-resolves on the next scroll/repaint. Not
  worth chasing further; flagging here so it isn't mistaken for a new regression if
  seen again during a demo.
- **Scripts used were throwaway** (`/tmp/extract_all_codes2.py`,
  `/tmp/verify_new_codes.py`, `/tmp/check_motor_codes.py`, `/tmp/motor_snippets.py`) —
  not committed to the repo, since they're one-off investigation tools, not part of
  the app. If this kind of full-manual code extraction needs to be repeated (e.g. when
  more manuals are added), the method is: `pypdf.PdfReader`, per-page
  `extract_text()`, regex `\b([FA]\d{5})\b\s*(\([A-Z]\)\s*)?([A-Z][^\n]{3,90})`, then
  the best-match-per-code scoring described above.

## 2026-08-19 — Roadmap divergence discovered against the official bootcamp spec

- **This project is a bootcamp final project, not a freestanding assignment.** The owner
  pointed to the official requirements PDFs at `C:\Users\se25479\Downloads\Requirements\`
  (`00_Overview_and_Project_Menu.pdf`, `09_The_Factory_Floor.pdf`) and asked for the
  internal `roadmap.pptx`/`README.md` build order to be checked against them before
  continuing to build. It hadn't been checked against the official spec since the
  original roadmap was drafted — real drift had accumulated.
- **The internal roadmap's "05 Computer Vision" description was wrong.** It said: "the
  user uploads a photo (nameplate or fault display), the system reads that image and
  extracts useful info (fault code, model)". The official spec's multimodal-core
  requirement is different: "classify the type of defect visible in the photograph — a
  scratch, a crack, contamination, and so on" — using the **MVTec Anomaly Detection**
  dataset, with a required classification-accuracy metric on a held-out test set. These
  are genuinely different features (defect-type classification vs. nameplate/code OCR) —
  caught before any code was written against the wrong interpretation, by re-reading the
  original spec instead of trusting the pptx.
- **Also found while cross-checking**: 3 of the 7 non-negotiable core requirements have
  no work done at all yet — Agents (tool-choosing + memory, not a fixed pipeline),
  Tracing/Observability (LangSmith/Langfuse/Arize — zero instrumentation), Evaluation with
  baseline benchmarking (`test_questions.txt` exists as a manual draft, never scored
  automatically, no baseline comparison at the RAG level). Plus a missing corpus type
  (Safety Data Sheets) needed for the "safety-first output contract" objective. All
  recorded in `README.md`'s new "Compliance checklist" section and
  `dificuldades_e_oportunidades.md` (items 8–11) — **not built this session**, by explicit
  owner instruction ("não quero alterar o escopo geral do projeto" — the scope was always
  this, per the original spec; only the tracking docs were wrong, and only the
  documentation gets touched now, not a decision to add new work).
- **Dataset decision for the vision path, confirmed with the owner**: MVTec AD has 15
  generic categories (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
  pill, screw, tile, toothbrush, transistor, wood, zipper) — most irrelevant to a
  motor/VFD copilot. Owner explicitly rejected using the full generic set ("não interessa
  nenhum dataset de outras coisas que não interessam para aqui"). Confirmed scope: a
  curated 4-category subset physically plausible on/around motors and VFDs — `cable`,
  `metal_nut`, `screw`, `transistor`. Searched for a motor/VFD-specific alternative first
  (no free dataset of comparable quality exists; closest is KolektorSDD, electrical
  commutators, but too small/narrow — 399 images, one defect type — to replace MVTec
  here).
- **Label vocabulary confirmed**: generic shared labels across all 4 categories (`good`,
  `scratch`, `deformation`, `structural_damage`, `contamination`, `other_defect`), not
  ~20 MVTec-native per-object labels (e.g. `cable__bent_wire`) — matches the spec's literal
  wording ("a scratch, a crack, contamination, and so on") and gives more training examples
  per class, at the cost of less per-photo specificity. Owner confirmed after a plainer,
  jargon-free re-explanation of the tradeoff (first phrasing of the question wasn't
  understood — don't assume ML vocabulary lands without a concrete example of what the UI
  would actually say back to the operator).
- **Feature-extraction approach confirmed**: frozen pretrained ResNet18 (`torchvision`) as
  a feature extractor + a small trained classifier on top (no fine-tuning — the bootcamp
  spec explicitly discourages fine-tuning), over lighter classical features
  (HOG/color-histogram). First genuinely heavy dependency added to this project
  (`torch`+`torchvision`); accepted deliberately for better expected accuracy.
- **`roadmap.pptx` editing convention going forward**: `python-pptx` is available in this
  environment (confirmed via `python -c "import pptx"`, no install needed). New slides are
  **appended**, never replacing/editing the two original hand-built diagram slides — the
  owner wants to review and validate roadmap changes before anything is considered final
  ("vais atualizar em slides novos, mantendo os atuais.. depois eu vou ver e valido, ou
  não"). Don't overwrite slide 1/2 without being asked again.
- **Full plan for the vision path (dataset prep, `factory_floor/defect_dataset.py`,
  `factory_floor/vision.py`, `notebooks/06_computer_vision.ipynb`, `app.py` integration,
  benchmarking) is written out in the approved plan file** — see the plan history for this
  session if picking this up fresh; the short version is in README's Compliance checklist
  and this note.
- **Built and verified end to end, same session**: `download_defect_images.py` (real MVTec
  AD images via the `Voxel51/mvtec-ad` Hugging Face mirror — the official mvtec.com archive
  URL is genuinely dead, confirmed 404), `factory_floor/defect_dataset.py`,
  `factory_floor/vision.py`, `notebooks/06_computer_vision.ipynb` (executed clean), and an
  `app.py` panel (file upload + "Classify defect" + optional "Compare with zero-shot LLM").
  Real results on the held-out test split: majority-class baseline 77.1%, zero-shot
  `gpt-4.1-mini` baseline 42.5% (worse than majority — it under-predicts `good`, a genuine
  failure-analysis finding worth using in the presentation), trained classifier (frozen
  ResNet18 features + LogisticRegression) 82.2%. Streamlit smoke-tested (headless launch +
  HTTP check, no tracebacks); all original RAG modules re-imported clean (zero regression).
- **Owner explicitly confirmed marking items 04 (Maintenance History) and 05 (Computer
  Vision) as Done**, after reviewing the corrected item 05 description. `roadmap.pptx`
  updated in place: item 04 status `On Going` → `Done`, item 05 status `Next` → `Done`,
  progress counter `44% / 4 of 9` → `67% / 6 of 9`. `README.md`'s point 10 and the
  Compliance checklist's multimodal-component row updated to match.
- **Known limitation of the pptx edit**: only the status *text* was changed. Each roadmap
  card also has a colored background tint + a colored number-badge circle tied to its
  status (green/amber/rust/grey, matching the legend), but the shape-to-status color
  mapping couldn't be verified reliably by reading the XML alone (indices didn't line up
  consistently between cards, and no LibreOffice was available in this environment to
  render the slide and check visually) — recompiling that risked recoloring the wrong
  shape. Left untouched rather than guess. If asked to fix this later: render the slide to
  an image first (e.g. install LibreOffice for `soffice --headless --convert-to png`) to
  confirm shape-to-color mapping before touching fills, don't infer it from shape index
  order alone.
- **`roadmap.pptx` editing needs the file closed first.** `python-pptx` raises
  `PermissionError` if PowerPoint has it open — happened twice this session. Check
  `Get-Process POWERPNT` before editing; if it's open, ask the owner to close it rather
  than force-killing (their unsaved view state isn't ours to discard).

## 2026-08-19 (cont.) — Vision path refined after live operator-facing feedback

- **Owner caught that a bare defect label is useless to an operator** ("o operador não vai
  usar isso para nada... apenas vai receber uma resposta a dizer se está em bom ou mau
  estado e indicações/sugestões de procedimentos"). Added
  `factory_floor/vision.py::recommend_actions(image, predicted_label, llm=None,
  language="English")` — one more LLM call, given the photo again plus the classified
  label, that returns concrete operator-facing next steps (safety precautions first).
  Deliberately **not** manual-grounded (no retrieval) — explicitly disclosed in both the
  prompt and the UI ("⚠️ Not grounded in this equipment's manuals..."). Fusing this with
  real manual citations is the Orchestrator Agent's future job, same scope boundary as
  before, not moved up.
- **Owner also caught that a photo submission didn't support conversation continuation**,
  unlike a text question. Fixed by unifying photo results into the same
  `st.session_state["turns"]` list text questions already use, tagged `"type": "photo"` vs.
  `"type": "text"` (default/implicit for `ask()`'s output). `render_turn()` now dispatches
  to `render_photo_turn()` for photo turns. A photo turn gets a synthetic
  `question`/`answer` pair (`"[Uploaded a photo...] Classified condition: X."` /
  the recommended-actions text) purely so `build_chat_history()` in `rag.py` — which reads
  `turn["question"]`/`turn["answer"]` unconditionally — keeps working unmodified for the
  text follow-up box that already existed. This gives light conversational continuity
  (the LLM's follow-up reformulation sees "a photo was analyzed, found X, recommended Y")
  **without** wiring photo analysis into RAG retrieval itself — retrieval for any follow-up
  question still only searches the manual vector store, exactly as before.
- **Also removed on request**: the "Classification detail" expander (probability bar
  chart, "Compare with zero-shot LLM" button) from the main app — owner said the operator
  has no use for it. That comparison logic still lives in `vision.py` and
  `notebooks/06_computer_vision.ipynb` (needed there for the required baseline-benchmarking
  evaluation), just not surfaced in the operator-facing UI anymore.
- **UI input redesign**: photo upload moved from its own always-visible section into the
  *same* initial form as the text question, joined by a plain "OR" divider and a single
  "Search manuals and answer" button (owner supplied a reference screenshot,
  `Downloads/test1.png`) — `submit_photo()` and `submit_question()` are chosen based on
  whether a photo was uploaded, mirroring the mockup exactly.
- **`roadmap.pptx` slide 3 (the checklist slide added earlier) updated**: row 4
  ("Multimodal component") status `In Progress` → `Done`, green, notes expanded with the
  real accuracy numbers (82.2% trained vs. 77.1% majority-class / 42.5% zero-shot
  baselines) and a mention of `recommend_actions()` + conversation-flow integration. Slide
  2's item 05 bullets were **not** re-edited this round — still accurate at the
  milestone-summary level; this refinement is an enhancement within the same milestone,
  not a new one.
- **New difficulties/opportunities logged** (`dificuldades_e_oportunidades.md`): #12
  (curated dataset is generic components, not real motor/VFD failure photos), #13 (the
  zero-shot baseline scores *below* the trivial majority-class baseline — a real, measured
  finding, not a hypothesis), #14 (`recommend_actions()` isn't manual-grounded).
  Opportunity #3 added per owner request: gather/curate real motor/VFD failure photos
  (worn bearings, burnt windings, corroded terminals, dusty VFD heatsinks, etc.) to replace
  or extend the generic MVTec subset.
- **Audio-based diagnostics explored, then explicitly deferred** — owner asked about
  diagnosing motor/VFD problems from recorded sound, same spirit as the vision path. Not
  required by the 7 core requirements (multimodal core already satisfied by vision), and
  the one genuinely domain-relevant free dataset found (UOEMD-VAFCVS, University of
  Ottawa — real induction motor + real VFD + real fault types, Mendeley Data DOI
  `10.17632/msxs4vj48g`) is small (128 recordings, multi-sensor CSV, needs real audio
  preprocessing) and not worth the time against the 5-week timeline right now. Full
  writeup in `dificuldades_e_oportunidades.md` opportunity #4 — don't re-research the
  dataset landscape from scratch if this gets picked up later, start from that note.

## 2026-08-19 (cont.) — Orchestrator Agent built (biggest of the 3 remaining core-requirement gaps)

- **Scope**: this is the "Agents" bootcamp core requirement — *"a model that decides which
  of several tools to use, uses them, and carries memory across the interaction."* Before
  this, `app.py` called `ask()` (RAG) or `classify_defect_trained()`+`recommend_actions()`
  (Vision) directly — fixed pipelines, zero decision-making.
- **Owner-confirmed design decisions before writing code** (asked via 3 plain-language
  questions, not decided unilaterally): (1) `recommend_actions()` is retired from the main
  flow — the agent itself produces the final recommendation now, and can ask clarifying
  questions whenever it has genuine doubt, not only on explicit signal conflict, per
  explicit owner request ("se tiver dúvidas, era fixe que ele próprio pudesse colocar
  questões"); (2) a photo is classified *before* the agent reasons (not an LLM-callable
  tool) — there's no real "whether" decision once a photo is uploaded, only "what to do
  with the result", which the agent still decides; (3) the form no longer forces "text OR
  photo" — both can be submitted together, since that's the only way to actually exercise
  conflict detection.
- **`langgraph` was already installed in this environment** (`langgraph==1.2.11`,
  `langgraph-prebuilt`, `langgraph-checkpoint`) but nothing in the project imported it
  before this — confirmed via `pip list` + grep, not assumed. Chosen over a manual
  `bind_tools()` loop because LangGraph traces show an actual agent→tools→agent graph
  (matches the "quality of agentic design, visible in traces" evaluation criterion
  directly) and wires up to LangSmith with just env vars — relevant since Tracing is the
  very next roadmap item.
- **Real gotcha caught during implementation, not just planned around**:
  `langgraph.prebuilt.create_react_agent` is deprecated in this environment's LangGraph
  version — a live `LangGraphDeprecatedSinceV10` warning during a smoke test pointed to
  the replacement, `langchain.agents.create_agent` (same underlying mechanics, `model` +
  `tools` + `system_prompt` instead of `prompt`). Switched before writing the real module,
  not after. Since `factory_floor/agent.py` only imports from `langchain.agents` and
  `langchain_core.tools` (not `langgraph` directly), `requirements.txt` did **not** need a
  new `langgraph`/`langgraph-prebuilt` line — `langchain>=1.3.0` already requires
  `langgraph` transitively (confirmed via `pip show langchain`).
- **`requirements.txt` had a real version-drift bug, fixed**: declared `langchain>=0.3.0`
  / `langchain-openai>=0.2.0` while the environment actually runs `langchain==1.3.15` (a
  different major series — `bind_tools`/`with_structured_output`/`create_agent` behave
  differently or don't exist across that gap). Minimums corrected to `>=1.3.0`/`>=1.5.0`
  etc. across the LangChain family, and `langchain-core` is now declared explicitly (it
  was already imported directly by `rag.py`/`vision.py`, just never declared).
- **Tool design, confirmed by testing, not just designed**: `search_manuals(query)` is
  deliberately thin (`retriever.invoke()` + `rag.format_context()`, no LLM synthesis
  inside the tool) so the agent does the final citing itself instead of re-wrapping an
  already-synthesized sub-answer — and because `format_context()` numbers `[SOURCE n]`
  from 1 on every call, which would collide if the agent calls the tool twice, the system
  prompt instructs citing by file+page instead (already present in each context block) —
  confirmed working in the smoke tests (the F30021 answer cited real page numbers, not
  transient source indices). `get_maintenance_history()` takes **zero** LLM-controlled
  arguments (`machine_id` closed over from the sidebar selection) — confirmed a zero-arg
  `@tool` works fine with `create_agent` via an isolated smoke test before writing it into
  the real module.
- **Memory**: no LangGraph `checkpointer` — reused the existing
  `st.session_state["turns"]` + `build_chat_history()` mechanism unchanged, passing
  `chat_history + [HumanMessage(...)]` per invocation, confirmed via a smoke test that a
  `SystemMessage` supplied per-call (not baked into `create_agent`) is honored correctly —
  this is what lets `language` vary per call without rebuilding the agent.
- **Verified with 4 real scenarios directly against `run_diagnostic_agent()`**, then again
  inside `notebooks/07_orchestrator_agent.ipynb` (executed clean via `nbconvert`) with
  `assert` checks on the two behaviors that matter most: (a) a real held-out
  `structural_damage`-classified photo paired with a text description downplaying it as
  "just a cosmetic scuff" → the agent asked a clarifying question instead of picking a
  side; (b) a `GENERAL` (no machine selected) question → `get_maintenance_history` was
  never even offered as a tool. Both confirmed programmatically, not just by reading the
  output.
- **`app.py` rewritten**: `submit_question()`/`submit_photo()` replaced by a single
  `submit_turn(question_text, uploaded_photo)` — accepts either, both, validates at least
  one is present. New unified turn shape (`"type": "agent"`) replaces the old
  `"text"`/`"photo"` split; `render_turn()` now shows a "Tools used" trace line per turn
  (e.g. "🔍 Searched manuals for: ...") so the operator — and anyone reviewing the demo —
  can see what the agent decided without opening LangSmith. The "⚠️ not manual-grounded"
  warning is now conditional on whether `search_manuals` actually appears in that turn's
  tool trace, not hardcoded to the photo-only path like before.
- **Streamlit background-process quirk observed again**: launching `streamlit run app.py`
  via a backgrounded PowerShell command occasionally reports `[exited with code 5]` a few
  seconds after a successful boot (confirmed via HTTP 200 + clean logs at the time), with
  no traceback anywhere. Happened at least twice this project. Root cause not identified —
  possibly the background-job wrapper's own lifecycle, not the Streamlit process itself
  (relaunching immediately after always works). Not worth chasing further unless it
  recurs with an actual error message attached.

## 2026-08-21 — Tracing, Safety Validator, and Evaluation (last 2 core requirements + roadmap item 06 closed)

Current milestone status update: **all 7 bootcamp core requirements now ✅ Done**, and all 9
internal roadmap items now Done (`roadmap.pptx` progress 78%/7-of-9 → 100%/9-of-9). Scope for
this session was agreed with the owner up front: Safety Validator is an audit utility used in
the evaluation notebook, **not** a live blocking gate in `app.py` (bigger scope, deferred); Git
repo init and the 3-page project doc are explicitly out of scope, left for a future session.

- **LangSmith project override has a real cache trap.** `langsmith==0.11.0`'s
  `langsmith.utils.get_env_var()` and `get_tracer_project()` are both
  `functools.lru_cache`'d. Setting `os.environ["LANGSMITH_PROJECT"]` after the *first*
  traced call happens is silently ignored — the cached value wins forever, with no error
  or warning. `factory_floor/tracing.py::configure_tracing()` calls
  `ls_utils.get_env_var.cache_clear()` and `ls_utils.get_tracer_project.cache_clear()`
  right after setting the env var, specifically because of this. Confirmed by testing:
  without the cache-clear calls, the override silently no-ops on the second+ traced call
  in the same process.
- **The override lives in `factory_floor/__init__.py`, not `config.py`.** `rag.py` never
  imports `config.py`, so a `config.py`-based override would be bypassed entirely by
  `from factory_floor.rag import ask` — the only module guaranteed to run on *any*
  `from factory_floor.X import Y` is `__init__.py`. The assignment there is
  **unconditional** (`os.environ["LANGSMITH_PROJECT"] = ...`, never `.setdefault(...)`)
  on purpose: `app.py` imports the package *before* calling `load_dotenv()`, while every
  notebook calls `load_dotenv()` *before* importing the package. `load_dotenv()` defaults
  to `override=False`, so an unconditional set wins under both import orders —
  `setdefault` would lose in the notebook case and leak traces back into the shared
  `.env`'s `lca-lc-foundation` project.
- **Getting a run URL for `@traceable`-decorated code needed a different mechanism than
  for `agent.invoke(config=...)` calls — discovered by testing, not assumed.**
  `LangChainTracer.get_run_url()` (via `tracing_v2_enabled()`) works for
  `run_diagnostic_agent()` (a LangGraph-compiled `.invoke()` call, goes through the
  callback-manager path) but raises `ValueError('No traced run found.')` for `rag.ask()`,
  since `ask()`'s internals use `@traceable` (langsmith's own RunTree-based tracing, a
  different mechanism the `LangChainTracer` instance never sees). `Client.read_run()`
  looked like the obvious fix but is a dead end: deprecated (removal after 2027-01-31)
  and requires `project_id`/`start_time` on this SmithDB-backed instance. The actual
  working, mechanism-agnostic fix: `Client._construct_run_url()` (via the public
  `client.get_run_url()` wrapper) only ever reads `run.id` off the object it's given —
  passing a bare `types.SimpleNamespace(id=run_id)` plus `project_name=` builds a correct
  permalink from *just* the `run_id` string every one of our functions already returns,
  with one cheap `read_project` call to resolve the project name to an id, no dependency
  on the run being independently queryable. This is `factory_floor/tracing.py::run_url()`
  — used identically after both `run_diagnostic_agent()` and `ask()` calls.
- **`rag.py::ask()` needed `@traceable`, not `config=` passed to each sub-call.** `ask()`
  is plain Python (`contextualize_question` → `retriever.invoke()` → `llm.invoke()`), not
  a single Runnable — passing a config to each of those would have produced 2-3 unrelated
  sibling root traces per call instead of one, which would have been a structurally unfair
  comparison against the agent's single grouped root run in the evaluation notebook.
  `@traceable(name="rag_baseline", run_type="chain")` on an inner `_ask_traced()` wraps the
  whole thing into one root run; `langsmith_extra={"run_id": ..., "tags": ..., "metadata":
  ...}` at the call site inside `ask()` supplies the same tagging story as the agent gets
  via `trace_config()`.
- **Rule 6 (safety-first) in `DIAGNOSTIC_SYSTEM_PROMPT` needed a "no action → no safety
  section" escape clause**, or it collides with rule 4 (ask a clarifying question on
  genuine doubt) and starts padding one-line answers with an unneeded safety block —
  which would have inflated the "zero failures" number by making the audit meaningless
  rather than by the system actually being safer. Also: **no literal `{`/`}` characters**
  in the new rule text — `DIAGNOSTIC_SYSTEM_PROMPT.format(language=...)` would raise on
  them.
- **The safety-first rule was deliberately never added to `rag.py`'s `SYSTEM_PROMPT`.**
  This was the single most important design decision behind the safety audit's
  credibility: if both pipelines had the rule, the audit's agent-vs-baseline contrast
  would just measure "does GPT follow instructions" twice, not "does this architectural
  choice matter." Measured result: baseline failure rate 94.1% (32/34 answers that
  recommended an action), agent (rule 6 active) 40.0% (12/30) — a real, large, honestly
  reported gap, not a hypothetical one.
- **The rule 6 prompt does not guarantee 100% compliance, confirmed on real API calls,
  more than once.** During `notebooks/08_tracing_observability.ipynb`'s very first live
  run (before the A/B test in notebook 09 was even written), the agent's F30059 answer put
  the safety section *after* the repair steps despite rule 6 already being active — caught
  by reading the actual printed output, not assumed to be correct. Confirmed again,
  independently, via a live Streamlit UI smoke test the same session (Playwright-driven,
  see below) — a real answer shown in the running app exhibited the same ordering failure.
  This is a genuine LLM instruction-following reliability limit at scale (37 diverse
  scenarios), not a bug to chase down within this session's scope — the audit's job is to
  *measure and report* this honestly, which it does (see the 40.0%/94.1% numbers above),
  not to guarantee zero via more prompt engineering. Don't be surprised or alarmed if a
  future rerun shows a similarly nonzero number; report it as the spec itself says to
  ("reporting it honestly is the point").
- **A safety-judge self-preference risk exists and is disclosed, not ignored**: the safety
  judge (`factory_floor/safety.py::check_safety_precautions`) is the same `gpt-4.1-mini`
  family the agent itself uses. `check_safety_precautions_keyword()` is a second,
  deterministic (regex-based) checker that exists specifically to cross-check the judge —
  measured agreement this session was 91.9% (agent answers) / 78.4% (baseline answers),
  printed in notebook 10 as an honest reliability signal on the judge's own number, same
  spirit as the zero-shot-vision-vs-majority-class disclosure from the Vision milestone.
- **`eval_scenarios.csv`'s evidence-keyword design specifically guards against parroting.**
  Every VFD shop-floor question in `test_questions.txt`/`eval_scenarios.csv` already
  contains the fault code *and* a symptom (e.g. "Tripped with F30003, DC link
  undervoltage..."), so a model that does nothing but echo the question would score a
  perfect keyword hit on a naive single-keyword check. Fixed by splitting into
  `expected_root_cause_keywords` (allowed to overlap the question — it's testing "stayed
  on the right fault") and `expected_evidence_keywords` (a phrase individually verified,
  same rigor as the 2026-08-18-d fault-code audit, against the real Cause/Remedy text of
  the retrieved manual chunk, and asserted absent from the question itself). A notebook-10
  bootstrap-cell assertion enforces this invariant on every row, not just at authoring
  time.
- **`history_reference_score()` had a real metric-leak bug, caught by a smoke test before
  it reached the notebook.** First version checked whether the answer mentioned either the
  machine's history event *date* or its *fault_code*. Since every eval question already
  names its own fault code, and several machines' simulated history happens to include an
  event with that same code, the baseline (`rag.ask()`, which has zero access to
  `machines.py`) scored `history_match=True` on scenario V01 purely by discussing the fault
  code the question already gave it — not because it referenced history at all. Fixed by
  checking *only* the event date (e.g. `"2022-06-25"`), which cannot appear in an answer by
  coincidence. Confirmed after the fix: baseline scores a clean 0.0% on this metric across
  all 37 scenarios, exactly as the structural invariant predicts.
- **Live Streamlit smoke test needed Python Playwright, not Node** — `node.exe` in this
  Windows sandbox produces **zero captured stdout/stderr**, confirmed several ways (bare
  `node --version`, redirecting to a file with `>`, wrapping in `cmd /c`), even though the
  process itself exits 0. This is a hard environment limitation, not a script bug — don't
  spend time debugging a Node driver script again in this environment; `pip install
  playwright` (the browser binary was already cached from a previous session at
  `%LOCALAPPDATA%\ms-playwright`, so no re-download needed) plus a plain Python
  `sync_api` script works fine and is now the established path here.
- **Streamlit's BaseWeb `st.selectbox` needs `wait_for_selector` + typed keyboard input,
  not an immediate click + `get_by_text`.** Two real issues, found by iterating against
  the live app rather than assumed: (1) the sidebar select renders as a grey skeleton for
  a couple of seconds while `load_rag_components()` builds the embeddings/Chroma
  connection on first load — clicking too early opens nothing. Fixed by waiting for the
  select element's `innerText` to be non-empty before interacting. (2) clicking the
  select to open it, then `get_by_text("VFD-06").click()`, timed out — the popover
  listbox item structure didn't match a plain text locator reliably. Fixed by clicking to
  focus the select, then `page.keyboard.type("VFD-06")` + `Enter`, which is how a real
  operator would narrow a BaseWeb select anyway.
- **Live UI verification confirmed real, working, end-to-end behavior**: selected VFD-06,
  asked about F30059, got a real grounded answer citing 3 real manual pages, no tracebacks,
  no console errors — the ordering-failure finding above was the only issue seen, already
  known and disclosed, not a new regression.

Full numbers: see `README.md`'s 2026-08-21 session note and `notebooks/10_evaluation_baseline.ipynb`'s
own printed output for the complete per-category and per-scenario breakdown.

## 2026-09-01 — Correções de higiene + fecho do ciclo de histórico

Sessão de análise do projeto que produziu três correções pequenas mas reais. Testes
164/164 verdes (`pytest -m "not llm"`), ruff limpo.

### Segredos — dois furos reais, tapados

- **`.env.bak.*` não estava no `.gitignore`.** O ignore cobria `.env` mas não os backups —
  um backup criado ao editar o `.env` ficava rastreável, e um `git add -A` distraído
  committava as chaves reais para um repo público. `.gitignore` ganhou `.env.bak`,
  `.env.bak.*` e `.env.local`; verificado que `.env.example` **não** casa com nenhum
  destes padrões e continua rastreado.
- **`Settings.openai_api_key` aparecia no `repr`.** Uma falha de teste em
  `test_config.py` imprimiu o início da chave real no output do pytest. Passou a
  `field(default=None, repr=False)`. Em CI não há chave, mas o `repr` do `Settings`
  aparece em tracebacks, logs e screenshots — não é sítio para um segredo.

### Isolamento de testes

`tests/conftest.py` carrega o `.env` real (de propósito, para os testes marcados `llm`
terem chave). Consequência: qualquer teste que afirme um **default** parte assim que o
dono configurar esse knob. Aconteceu — pôr `FACTORY_FLOOR_DAILY_SPEND_CAP_USD=5.00` no
`.env` partiu `test_optional_float_parsing`, que assumia `None`. Corrigido com um
`monkeypatch.delenv` na própria assertion (o padrão que o teste vizinho
`test_from_env_with_nothing_set_equals_defaults` já usava). **Ao escrever um teste novo
sobre um default de `Settings`, limpa a env var primeiro** — não confies no `.env`.

### Dificuldade 18 fechada — o agente vê agora as resoluções dos operadores

Descoberta pelo dono em teste manual: gravar uma resolução para `F30805`/VFD-04 e voltar
a perguntar o mesmo chamava o agente na mesma, ignorando o que tinha acabado de ser
gravado. Causa: `build_history_tool` chamava `get_machine_history(machine_id)` com o
default `include_resolutions=False` (escolha da fase 5, para não mexer no notebook 05).

- Agora `include_resolutions=True` na tool do agente. O notebook 05 usa
  `machines.get_machine_history` diretamente e fica inalterado.
- `format_history()` ganhou `max_rows=HISTORY_MAX_ROWS` (20) + ordenação por data. **Isto
  não é cosmético:** os `operator_resolution` acumulam sem limite, e sem cap o prompt do
  agente crescia a cada turno para sempre.
- O `DIAGNOSTIC_SYSTEM_PROMPT` distingue agora `[operator_resolution]` como nota de campo
  **não verificada** — pista, não facto documentado, nunca citável como manual. (Lembrar:
  o prompt passa por `.format(language=...)`, por isso **nada de `{` ou `}` literais**.)

### Dificuldade 19 (nova) — o gate gastava sem ser contado

O `CostTrackingCallback` só era anexado à chamada do agente. O juiz de segurança (+
reescrita + re-verificação, até 3 chamadas) era gasto real invisível à linha de custo da
sidebar **e ao teto diário** — o teto podia ser ultrapassado sem bloquear.
`check_safety_precautions()` e `enforce_safety()` aceitam agora `config` opcional,
reencaminhado para os três `.invoke()`; `services._apply_gate` passa o mesmo
`agent_config`. Coberto por `TestCostConfigForwarding`.

### Ainda por fazer (decisões do dono, não bugs)

- **Oportunidade 5** — atalho de custo-zero quando já existe resolução para
  `(máquina, código)` exato. Isto fecha o ciclo de *custo*; a 18 fechou só o de
  *conhecimento* (o agente sabe, mas continua a pagar uma chamada).
- **`OP-3001` (nome real do dono) está no `operators.csv` alterado.** O hash não se
  reverte, mas um PIN de 4 dígitos parte-se por força bruta em segundos — decidir se
  entra num repo público.
- **Reranker sem knob** — `services.build_diagnostic_retriever` fixa `rerank=True`. É a
  chamada LLM mais "opcional" do caminho crítico; um `FACTORY_FLOOR_RERANK_ENABLED`
  daria uma alavanca de latência sem mexer em código.

### Ferramenta nova: `manage_operators.py`

Gerir operadores à mão no `operators.csv` é impossível (o PIN é hash PBKDF2, não texto).
`manage_operators.py` na raiz faz `list` / `add` / `setpin` / `remove`. PINs de demo
committados: OP-1001 `1234`, OP-1002 `5678`, OP-2001 `4321`.

## 2026-09-01 (cont.) — Painel de ocorrências anteriores, a custo zero

O dono perguntou, com o histórico do VFD-04 à frente: *"este defeito já tinha acontecido, não devia
ter corrido o modelo?"*. A investigação deu três respostas, e uma funcionalidade nova.

### Porque é que o modelo correu

1. **O agente nem chamou a ferramenta de histórico.** O `tool_calls` do turno mostrava só
   `search_manuals`. É um agente — decide — e para um código puro achou que o manual chegava.
2. **Mesmo que tivesse chamado, o LLM corria na mesma.** A ferramenta devolve texto cru; ler e
   compor a resposta *é* uma chamada. Ter histórico informa, não salta o modelo.
3. **Não havia por onde procurar.** `resolution_events` não tinha `fault_code` — as resoluções
   gravadas apareciam com o código vazio. Era o bloqueio real da oportunidade 5.

### A armadilha que evitámos (importante para a defesa)

Com os dados reais: `F30805` no VFD-04 foi respondido **2× com "Replaced the power unit module" e
voltou na mesma**, depois 3× com desliga-liga. **Uma recomendação por frequência escolheria o
desliga-liga, 3 votos a 2** — recomendaria mascarar uma avaria de hardware recorrente. Sem saber se
cada ação *funcionou*, "mais frequente" ≠ "mais eficaz". Daí o desenho: **relatar, nunca
recomendar**, e recolher `outcome` para que um dia se possa.

### O que existe agora

- **`factory_floor/recurrence.py`** — sem LLM nenhum. Contagem, ordenação e datas sobre registos
  existentes. `RECURRENCE_WINDOW_DAYS = 28` é um juízo de manutenção, não um limiar medido — é uma
  bandeira para o operador ler, nunca um veredicto automático.
- **Interceção em `app.py::submit_turn`**, no mesmo padrão do `pending_typo`: `pending_prior` +
  `prior_occurrence_ack`. Só dispara em **primeira pergunta, sem foto, com um único código, com
  máquina selecionada e com histórico desse código**. Interromper um follow-up quebraria a linha de
  raciocínio em que o operador já está.
- **`code_in_question()` devolve None com dois códigos** — "esta avaria já aconteceu?" não tem
  resposta única quando há duas avarias na pergunta.
- **`resolution_events` + `fault_code` + `outcome`**, com migração `ALTER TABLE` idempetente
  guiada por `PRAGMA table_info` (`_ADDED_COLUMNS`). **Armadilha:** o índice sobre `fault_code` teve
  de sair do `_SCHEMA` para um `_SCHEMA_AFTER_MIGRATION` — numa BD antiga o `executescript` corre
  *antes* dos `ALTER` e rebenta com "no such column".
- Teste `test_never_recommends_an_action` falha se alguém acrescentar uma recomendação ao resumo.

### Verificado ao vivo

VFD-04 + "F30805 appeared again this morning" → painel com as 2 ocorrências, aviso "returned
before — shortest gap 1385 days", e os botões de escolha. **A barra de custo não mexeu** — zero
chamadas ao modelo.

## 2026-09-03 — README reescrito para portefólio; histórico movido para `docs/project_log.md`

O repositório é a peça de portefólio para candidaturas, e o `README.md` ainda era o
diário de trabalho: 602 linhas, parado a 2026-08-25, com secções tipo *"Session note —
read this FIRST before continuing"* e *"What to show in the Saturday presentation"*, e
sem uma palavra sobre nada da professionalização (service layer, custo, gate de
segurança, audit trail, identidade, cache, tenancy, API, testes) nem sobre o trabalho de
01-09 (gate multilingue, painel de ocorrências).

**O que mudou:**

- **`README.md` reescrito de raiz** como front door: o que é → o que uma consulta faz
  passo a passo → arquitetura (diagrama atualizado, com o agente, o gate, o custo, o
  audit e o `services.py` como seam) → decisões de engenharia com os números medidos
  (códigos de falha 18%→100%, o bug do gate em 4 das 5 línguas, o painel de recorrência
  que relata e recusa recomendar, o reranker, agent-vs-baseline incluindo os números
  que não favorecem o agente) → tabela de production concerns → quickstart + knobs →
  dados (corpus, frota, os 17 códigos num `<details>`) → estrutura → limitações →
  background do bootcamp → licença.
- **`docs/project_log.md` (novo)** — todas as session notes datadas, a checklist de
  conformidade dos 7 requisitos core e o guião da apresentação, **verbatim**. Nada foi
  deitado fora; o README só deixou de ser o sítio onde isso vive. As referências deste
  ficheiro a "README's 2026-08-25 note" e afins apontam agora, na prática, para lá.
- **`LICENSE` (novo)** — MIT, `Copyright (c) 2026 Marcelo Correia`. O repo não tinha
  licença nenhuma, o que num repo público significa "todos os direitos reservados".
- **Ficheiros destrackados**: `app_err.log` (78 KB), `app_out.log` e
  `.claude/settings.local.json`. Verificado antes de os tirar: os logs **não continham
  segredos** (zero matches de `sk-`, `api_key` ou paths locais) — era só ruído de
  execução do Streamlit. `.gitignore` ganhou `*.log` e `.claude/settings.local.json`.

Verificado antes do commit: **183 testes passam**, ruff limpo.

**Ainda por fazer, e só o dono pode:** o repositório continua **privado** — a API pública
do GitHub devolve 404 nesse URL. Enquanto assim for, o link não mostra nada a quem o
receber, e o badge de CI no topo do README também não renderiza. Tornar público em
Settings → General → Danger Zone → Change visibility.
