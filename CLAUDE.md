# The Factory Floor — working notes for Claude

Read this before changing anything. It holds the rules, traps and standing decisions a
session needs and nothing else. Keep it that way: when something here stops being true,
fix the line; when a session produces history worth keeping, append it to
`docs/project_log.md`, not here.

## What this is, and where to read

Industrial maintenance copilot (RAG + LangGraph agent + vision + live safety gate) for
Siemens motors and VFDs. Bootcamp final project, all requirements met; now a portfolio piece.

- `README.md` — what the system is and the measured numbers (the front door).
- `docs/project_report.pdf` — the 8-page write-up; source HTML in
  `~/Desktop/Factory_Floor_arquivo/deck_build/report/` (outside the repo).
- `docs/limitations_and_opportunities.md` — what is still open, plus a table of what was closed.
- `docs/project_log.md` — every dated session note since 2026-08-18, verbatim, including the
  1,100 lines this file used to carry. Look there before re-deriving any past decision.
- `docs/backend_architecture.md`, `docs/multi_tenancy.md`, `docs/secrets.md` — designs for
  what has a seam but no implementation.

## Environment

- Python is the conda `base` env: `/opt/miniconda3/bin/python` (3.13). The `python3` on PATH
  (3.14) is a partial install. Run `make test PYTHON=/opt/miniconda3/bin/python`, same for
  `make lint`; notebooks with `python -m nbconvert --to notebook --execute --inplace`.
- `langchain_openai` imports slowly (6–19 s, pulls in torch); uvicorn cold start is ~20 s.
- `.env` is a shared personal file with keys for services this project does not use. It is
  kept as-is on purpose (owner: "não separes o .env, sei o que faço"). Never split or clean it.
  Local-only knobs currently on: `REQUIRE_LOGIN`, daily cap 5 USD, semantic cache.
- `data/` (~1.9 GB: manuals, vector stores, images, audit DB) is gitignored. Notebooks 01/02
  rebuild `data/vectorstore/` — never run them casually, the demo depends on that store.

## Architecture invariants

- `factory_floor/services.py::run_diagnostic` is the seam. Cache lookup → spend-cap check →
  agent → safety gate → audit row → cache store all happen there. `app.py` reads widgets and
  renders; `api/main.py` does JSON; neither owns pipeline logic. Add behaviour in the seam.
- Every hardening feature is opt-in via `FACTORY_FLOOR_*` env vars (`config.py::Settings`,
  documented in `.env.example`). A fresh clone with only `OPENAI_API_KEY` must behave like the
  original RAG app. Exception already made: the safety gate defaults to `rewrite`.
- Streaming contract: `run_diagnostic(stream=True)` returns `(generator, result)`; the result
  fields are empty until the generator is fully consumed. With the gate on, the stream is
  drained and the gated text is emitted as one chunk.
- Fault codes are looked up literally (`fault_codes.py` + `CodeAwareRetriever`, Chroma
  `where_document $contains`), pinned first, never reranked. Semantic search handles the rest.
- Small hand-made reference tables live at repo root as CSV (`machines.csv`,
  `maintenance_history.csv`, `operators.csv`, `fault_codes.csv`, `manual_sources.csv`,
  `eval_scenarios.csv`, `tenants.csv`); bulk generated artefacts live in `data/`.

## Traps that already bit (one line each; details in the log)

- `get_settings()` is `lru_cache`d and first runs at package import, before `load_dotenv()` in
  `app.py`/`api/main.py` — both call `get_settings.cache_clear()` after loading `.env`.
- LangSmith's `get_env_var`/`get_tracer_project` are `lru_cache`d: `tracing.py` clears them
  after setting `LANGSMITH_PROJECT`. That env var is set unconditionally in
  `factory_floor/__init__.py` so it wins under both import orders (app vs notebooks).
- `DIAGNOSTIC_SYSTEM_PROMPT` goes through `.format(language=...)` — no literal `{` or `}`.
- `tests/conftest.py` loads the real `.env` (for `llm`-marked tests). Any test asserting a
  `Settings` default must `monkeypatch.delenv` that var first.
- SQLite migration order: indexes on columns added by `ALTER TABLE` live in
  `_SCHEMA_AFTER_MIGRATION`, not `_SCHEMA`, or an old DB fails with "no such column".
- The main Chroma collection is `hnsw.space='l2'`: LangChain's relevance scores fall outside
  0–1 there, so `similarity_score_threshold` is meaningless on it. The semantic cache uses its
  own cosine collection for that reason.
- Any language-keyed optimisation is a feature switch in disguise: the gate's "no action verb →
  skip the judge" shortcut used English verbs only and silently disabled the gate in 4 of 5
  languages. The shortcut is now English-only; everything else always goes to the judge.
- The cost callback must be passed to every LLM call, including the gate's judge/rewrite/
  re-check (`config=` on `enforce_safety`), or spend escapes the sidebar line and the daily cap.
- Fake models: `GenericFakeChatModel` has no `bind_tools`/`with_structured_output`; use
  `conftest.make_agent_fake_llm` / `make_structured_llm` / `make_gate_llm`.
- Streamlit: any button click reruns the script (state goes in `st.session_state`);
  `st.columns()` anchors content where it was called, not where it is re-entered; to clear a
  widget change its `key` (see `followup_key`), `del` + rerun is unreliable.
- `format_context()` numbers `[SOURCE n]` from 1 on every call, so the agent cites by file +
  page, not by source number.

## Decisions not to "fix" without asking

- Prior-occurrence panel (`recurrence.py`) reports and never recommends: on real data a
  frequency ranking would recommend masking a recurring hardware fault. A test guards it.
- Unknown fault code → clean refusal, and do **not** list similar codes. A display typo
  (`F3OO21`) → ask the operator, never auto-correct, never feed the guess to the LLM.
- Rule 6 (safety-first) is deliberately absent from the baseline `rag.py::SYSTEM_PROMPT`;
  otherwise the agent-vs-baseline safety audit measures "GPT obeys" twice.
- `machines.get_machine_history` defaults to `include_resolutions=False` (notebook 05 depends
  on it); the agent's tool passes `True`. `format_history` caps at 20 rows, oldest first.
- `operators.csv` holds only the 3 fictional operators (PINs 1234 / 5678 / 4321). The owner's
  real ID was removed on purpose; a 4-digit PIN is brute-forceable in a public repo.
- Photo is classified before the agent reasons (not an LLM-callable tool). The agent produces
  the final recommendation; `vision.recommend_actions` survives only for notebook 06.
- The recurrence panel only intercepts the first question of a conversation, with one code,
  no photo, a machine selected and history for that code. Interrupting a follow-up would break
  the operator's line of thought.
- `download_manuals.py` was not the path used to fetch the PDFs, but it is live-tested and is
  the documented reproducibility step.
- The reranker is hard-coded `rerank=True` in `services.build_diagnostic_retriever` — the knob
  is opportunity 6, not yet built.

## Testing convention

- Per commit: `make test` (183 tests, no key, ~free) + `make lint`. CI runs both on 3.12/3.13.
- At a phase boundary (before merging): re-execute notebooks 03–14 with nbconvert and drive the
  app in a real browser. Both cost real API money; 01/02 are never re-run.
- Demo prep: `FACTORY_FLOOR_REQUIRE_LOGIN=true` means sign in first; VFD-06 has F30059 history
  so the recurrence panel intercepts that question; clear the semantic cache before presenting
  or the answer comes back as "⚡ Answered from the cache" at $0.
- Numbers quoted in `README.md` and `docs/project_report.pdf` must match the printed output of
  notebooks 06, 10 and 11. When a notebook is re-run, re-check both documents.

## Open items

See `docs/limitations_and_opportunities.md`. The short list: no relevance floor (needs a cosine
rebuild), keyword scoring is a proxy, 2 of 34 answers still fail safety ordering after the gate,
vision data is not domain data, no SDS corpus, reranker knob, audio (dataset identified).
