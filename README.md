# The Factory Floor

**An industrial maintenance copilot for electric motors and variable-frequency drives.**
A technician standing in front of a tripped Siemens SINAMICS drive asks a question in their own
language, optionally with a photo, and gets an answer grounded in the real manufacturer manuals —
with page citations, safety precautions first, that machine's own repair history, and a running
cost meter.

[![CI](https://github.com/mcorreia10/Factory_Floor_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/mcorreia10/Factory_Floor_Chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Tests](https://img.shields.io/badge/tests-183%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Built on **11 official Siemens manuals (3,679 pages)** covering SINAMICS G120/G120C drives and
SIMOTICS GP/SD motors, over a simulated fleet of **20 assets** with their own maintenance history.

**Full write-up:** [`docs/project_report.pdf`](docs/project_report.pdf) (8 pages) ·
**Slides:** [`docs/Final_Presentation.pptx`](docs/Final_Presentation.pptx)

> **Scope, stated plainly:** decision support for a trained technician, built as a portfolio
> project. Not a certified industrial safety system. See [Limitations](#limitations).

---

## What a diagnostic turn does

1. **Operator signs in** (optional, PIN-hashed) and picks a machine — that selection filters
   retrieval to its equipment type and unlocks its history.
2. **Prior occurrences appear before any LLM runs.** If the question carries a fault code this
   machine has seen before, a zero-cost panel shows when, what was done, and whether it came
   back — computed from records, never generated.
3. **The agent decides** which tools to use: `search_manuals` (code-aware + reranked retrieval),
   `get_maintenance_history`, plus any photo classification injected as context — or it asks a
   clarifying question instead of guessing.
4. **A safety gate audits the answer before the operator sees it.** A physical action without
   precautions first is rewritten (or blocked) rather than shown.
5. **Everything is metered and recorded**: token cost against a daily cap, and an audit row with
   the recommendation, its sources, the tools used and the operator who asked.
6. **The operator writes back what actually fixed it**, with an outcome, which feeds the
   machine's history and the next diagnosis of the same fault.

Five UI languages (EN / FR / PT / ES / DE), because the shop floor is not monolingual.

---

## Architecture

```text
  Official Siemens PDFs (11 manuals, 3,679 pages)
                    │
              ingestion.py  ── page + equipment_type metadata
                    │
             vectorstore.py ── chunking (800/120), embeddings, persistent Chroma (12,301 chunks)
                    │
   ┌────────────────┴─────────────────┐
   │  retrieval                       │
   │   • CodeAwareRetriever  (literal fault-code lookup, no embeddings)
   │   • RerankingRetriever  (15 candidates → one LLM reordering call → 5)
   └────────────────┬─────────────────┘
                    │
  ┌─────────────────┴──────────────────────────────────────────┐
  │  agent.py — LangGraph diagnostic agent (gpt-4.1-mini)      │
  │    tools: search_manuals · get_maintenance_history         │
  │    context: selected machine · photo classification        │
  │             · operator-written past resolutions            │
  └─────────────────┬──────────────────────────────────────────┘
                    │
             safety.py ── live gate: rewrite / block / off
                    │
    ┌───────────────┼───────────────┬──────────────┐
 cost.py        audit.py        cache.py     recurrence.py
 (meter +      (SQLite trail,  (opt-in       (prior occurrences,
  daily cap)    resolutions,    semantic      zero LLM calls)
                CMMS export)    cache)
    └───────────────┴───────────────┴──────────────┘
                    │
            services.py ── the seam: one diagnostic turn, no Streamlit, no HTTP
                    │
        ┌───────────┴───────────┐
     app.py                  api/main.py
   (Streamlit UI)        (FastAPI + Dockerfile)
```

`services.run_diagnostic()` is the seam everything wraps. The Streamlit app reads widgets and
renders; the API does JSON. Neither owns any pipeline logic.

**Stack:** LangChain 1.3 · Chroma · `text-embedding-3-small` · `gpt-4.1-mini` at temperature 0 ·
`langchain.agents.create_agent` (LangGraph) · torchvision ResNet18 + logistic regression ·
LangSmith · Streamlit · FastAPI · pytest + ruff + GitHub Actions.

---

## What the numbers say

Every figure is the printed output of a notebook (`06`, `10`, `11`), as of 2026-09-04.

| Measurement | Result |
|---|---|
| **Fault-code queries that bring the defining page** (17 codes, code alone) | semantic search **18%** → literal lookup **100%**; `F99999` refuses, `F3OO21` asks |
| **Defect classifier** (376 held-out MVTec images, 4 categories) | trained **82.2%** · majority-class 77.1% · zero-shot LLM **42.5%** (worse than doing nothing) |
| **Root-cause accuracy**, 37 scenarios (agent / single-call baseline) | **75.7%** / 73.0% |
| **Evidence accuracy** (verified manual phrase present) | **54.1%** / 45.9% |
| **Machine-history awareness** | agent cites the real event; baseline structurally 0% |
| **Conflicting signals** (photo says damage, text says "cosmetic scuff") | agent asks a question; baseline says *"likely safe to continue running"* |
| **Safety-ordering failures** — baseline / agent raw / agent **delivered after the gate** | 91.2% / 33.3% / **5.9%** (2 of 34, named, not tuned away) |
| **Retrieval ablation** (hit-rate@5 / MRR) | chunk 800 → 78.4% / 0.596 · reranker off → on 81.1% → **86.5%**, MRR 0.610 → **0.695**, on half the context of plain k=10 |
| **Observability** (LangSmith, project lifetime) | 1,620 runs · 0% exceptions · P50 1.71 s · P99 8.98 s |

Four things behind those numbers, each found by measuring, not by design:

- **Fault codes are looked up, not searched.** Embeddings represent meaning and `F30021` has
  none; Chroma's own full-text filter (`where_document $contains`) does the exact match with no
  new dependency. Exact hits are pinned first and never reranked.
- **The safety gate was silently off in 4 of 5 languages.** The "no physical action → skip the
  judge" shortcut used English verbs only. Found by chasing *"why is the Portuguese answer nothing
  like the English one?"*, not by reading the code.
- **The prior-occurrence panel reports and refuses to recommend.** On real data VFD-04's `F30805`
  was "fixed" twice by a module swap (it came back) and three times by a power cycle; a frequency
  ranking would recommend masking a hardware fault 3-to-2.
- **Two features were built, measured and reverted** — a thermal-image dataset (369 images, only
  11 independent scenes) and a component-identity classifier (100%, but every category had its
  own background). Both are written up in
  [`docs/limitations_and_opportunities.md`](docs/limitations_and_opportunities.md).

Full ablation and evaluation detail: [`Retrieval_Benchmark_Report.pdf`](Retrieval_Benchmark_Report.pdf),
`notebooks/10_evaluation_baseline.ipynb`, `notebooks/11_retrieval_benchmark.ipynb`.

---

## Production concerns, built in

Everything here is **opt-in and off by default** — a fresh clone with only `OPENAI_API_KEY`
behaves exactly like the original RAG app.

| Concern | How it is handled |
|---|---|
| **Configuration** | Typed `Settings` dataclass (`config.py`), every knob a `FACTORY_FLOOR_*` env var with a safe default |
| **Secrets** | `secrets.py` is a seam — `env` today, vault backends designed in [`docs/secrets.md`](docs/secrets.md). The key never appears in a `repr` |
| **Cost** | Per-token metering, per-session total, a daily ledger and a hard spend cap — including the safety gate's own LLM calls |
| **Safety** | Live gate: `off` \| `rewrite` \| `block`, applied before the operator sees the answer |
| **Auditability** | SQLite (WAL) trail of every recommendation, its sources, tools, cost and operator; operator-written resolutions with outcome; CMMS-export demo; email hand-off |
| **Identity** | `operators.csv` with PBKDF2-hashed PINs, managed via `manage_operators.py` (never hand-edited) |
| **Caching** | Opt-in semantic answer cache (Chroma, cosine), exact-match only for fault codes |
| **Multi-tenancy** | Collection-per-tenant seam in `tenancy.py`; design in [`docs/multi_tenancy.md`](docs/multi_tenancy.md) |
| **Serving** | FastAPI proof (`/health`, `POST /diagnose`) + Dockerfile; full surface in [`docs/backend_architecture.md`](docs/backend_architecture.md) |
| **Tests / CI** | 183 tests (unit + integration, no API key, no cost), ruff, GitHub Actions on Python 3.12 and 3.13 |
| **Tracing** | LangSmith project routing + permalink per run (`tracing.py`) |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                 # add your OPENAI_API_KEY
python download_manuals.py                           # 11 official Siemens PDFs → data/manuals/
python download_defect_images.py                     # MVTec AD subset → data/defect_images/  (vision only)
```

Run `notebooks/01_data_ingestion.ipynb` and `notebooks/02_vector_database.ipynb` once to build
the Chroma store under `data/vectorstore/`. Then:

```bash
make app          # streamlit run app.py
make test         # 183 tests, no API key needed, no cost
make lint         # ruff
uvicorn api.main:app --reload        # or the API: /health, POST /diagnose
docker build -t factory-floor . && docker run -p 8000:8000 -v "$PWD/data:/app/data" factory-floor
```

`make` targets: `test` · `test-all` · `test-llm` · `lint` · `fmt` · `nb` · `app`. Override the
interpreter with `make test PYTHON=/path/to/python`. **Notebooks 01–14** walk the whole build in
order: ingestion, vector DB, RAG, conversation history, maintenance history, vision, the agent,
tracing, the safety validator, evaluation, the retrieval ablation, the service layer, the audit
trail, and the semantic cache.

### Configuration

All optional, all documented in `.env.example`, all defaulting to the original behaviour:

| Knob | Default | Effect |
|---|---|---|
| `FACTORY_FLOOR_DAILY_SPEND_CAP_USD` | unset | Hard daily cost ceiling |
| `FACTORY_FLOOR_SAFETY_GATE_MODE` | `rewrite` | `off` \| `rewrite` \| `block` |
| `FACTORY_FLOOR_REQUIRE_LOGIN` | `false` | Operator sign-in before asking |
| `FACTORY_FLOOR_AUDIT_ENABLED` | `true` | SQLite audit trail |
| `FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED` | `false` | Reuse near-identical questions at $0 |
| `FACTORY_FLOOR_SECRETS_BACKEND` | `env` | Vault backends are design-only seams |
| `FACTORY_FLOOR_TENANT_ID` | `default` | Collection-per-tenant routing |

---

## The data

- **Corpus — 11 official Siemens manuals, 3,679 pages**, five document types (operating
  instructions, list/fault manuals, function manuals, engineering manual, catalog): SINAMICS
  G120 / G120C / CU240B-2 / CU240E-2 on the drive side, SIMOTICS GP / SD / DP / 1LE1 / 1LE7 on
  the motor side. ~2,766 pages VFD / ~913 motor. `manual_sources.csv` carries an explicit
  `equipment_type` per manual and the official download URL.
- **Fleet — 20 simulated assets** (10 motors + 10 VFDs) in `machines.csv`, with 77 fault /
  repair / preventive events in `maintenance_history.csv`, generated once with a fixed seed.
- **Fault codes — 368 real codes** in `fault_codes.csv`, produced by `build_fault_code_index.py`
  from a regex scan of every page of raw PDF text — not invented, not sampled from a semantic
  search. The 17 codes used in the fleet history and the evaluation set were each verified
  against the real Cause/Remedy text. Electric motors have no numeric fault-code system at
  all (zero matches across all five motor manuals); motor events use the manuals' own wording.
- **Evaluation — 37 scenarios** in `eval_scenarios.csv`, each with a verified root cause and a
  manual evidence phrase confirmed absent from the question, so nothing scores by echo.
- **Defect images — MVTec AD**, curated to the 4 categories plausible on a motor or drive
  (cable, metal nut, screw, transistor): 1,502 images, 1,126 train / 376 held out.

---

## Project structure

```text
├── app.py                  # Streamlit UI — a thin view over services.run_diagnostic()
├── api/main.py             # FastAPI proof: /health, POST /diagnose  (+ Dockerfile)
├── factory_floor/
│   ├── services.py         # THE SEAM: one diagnostic turn, no UI, no HTTP
│   ├── agent.py            # the diagnostic agent — tools, safety-first contract
│   ├── rag.py · fault_codes.py   # retrievers, prompts, literal fault-code lookup
│   ├── safety.py           # post-hoc audit + live blocking gate
│   ├── audit.py · cost.py · cache.py · recurrence.py · identity.py · tenancy.py
│   ├── vision.py · defect_dataset.py · evaluation.py · retrieval_benchmark.py · tracing.py
│   └── config.py · secrets.py · ingestion.py · vectorstore.py · machines.py · manuals.py
├── tests/                  # 183 tests — unit + integration, no API key, no cost
├── notebooks/              # 01–14, the build in order
├── docs/                   # project_report.pdf · Final_Presentation.pptx · designs · project_log.md
├── *.csv                   # machines, history, operators, tenants, manual sources, fault codes, eval scenarios
└── data/                   # gitignored: manuals, vector stores, images, audit DB
```

---

## Limitations

- **Not a certified safety system.** The gate reduces precaution-ordering failures from 33.3%
  to 5.9%; it does not eliminate them, and the two residual cases are published, not tuned away.
- **The fleet and its maintenance history are simulated.** The manuals, the fault codes and every
  measured number are real.
- **Vision runs on MVTec AD**, not photographs of motors. Transfer to a real photo inside a
  drive cabinet is explicitly not established.
- **Evidence accuracy (54.1%) is the weakest number**, and the keyword scorer behind it is a
  proxy. A relevance floor cannot be tested until the store is rebuilt in cosine space.
- **Designed but not built:** a real vault backend, live multi-tenant deployment, the full API
  surface with streaming, compose + load balancing. Each has a design doc and a seam.

Open gaps, abandoned experiments and the resolved-items table:
[`docs/limitations_and_opportunities.md`](docs/limitations_and_opportunities.md).
The dated build history: [`docs/project_log.md`](docs/project_log.md).

## Background

Originally the final project for the Ironhack AI Engineering bootcamp — all seven core
requirements (RAG with citations, agents, tracing, a multimodal component, a ≥30-task evaluation
set with baseline benchmarking, web deployment, presentation) were met. Everything above the
bootcamp line — the service layer, cost control, the live safety gate, the audit trail, identity,
caching, tenancy seams, the API and the test suite — was built afterwards to take it from a
working demo to something that could survive contact with a real deployment.

## License

MIT — see [LICENSE](LICENSE).
