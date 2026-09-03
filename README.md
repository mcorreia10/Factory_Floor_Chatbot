# The Factory Floor

**An industrial maintenance copilot for electric motors and variable-frequency drives.**
A maintenance technician standing in front of a tripped Siemens SINAMICS drive asks a question
in their own language, optionally with a photo, and gets an answer grounded in the real
manufacturer manuals — with page citations, safety precautions first, that machine's own repair
history, and a running cost meter.

[![CI](https://github.com/mcorreia10/Factory_Floor_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/mcorreia10/Factory_Floor_Chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Tests](https://img.shields.io/badge/tests-183%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Built on a corpus of **11 official Siemens manuals (3,679 pages)** covering SINAMICS G120/G120C
drives and SIMOTICS GP/SD motors, over a simulated fleet of **20 assets** with their own
maintenance history.

> **Scope, stated plainly:** this is a decision-support tool built as a portfolio project, not a
> certified industrial safety system. See [Limitations](#limitations).

---

## What a diagnostic turn actually does

1. **Operator signs in** (optional, PIN-hashed) and picks a machine from the fleet — that
   selection filters retrieval to that equipment type and unlocks its history.
2. **Prior occurrences appear before any LLM runs.** If the question carries a fault code this
   machine has seen before, a zero-cost panel shows when, what was done, and whether it came
   back — computed from records, never generated.
3. **The agent decides** which tools to use: `search_manuals` (code-aware + reranked retrieval),
   `get_maintenance_history`, plus any photo classification injected as context.
4. **A safety gate audits the answer before the operator sees it** — if a physical action is
   recommended without precautions first, the answer is rewritten (or blocked) rather than shown.
5. **Everything is metered and recorded**: token cost against a daily cap, and an audit row with
   the recommendation, its sources, the tools used and the operator who asked.
6. **The operator writes back what actually fixed it**, which feeds both the machine's history
   and the next diagnosis of the same fault.

Five UI languages (EN / FR / PT / ES / DE), because the shop floor is not monolingual.

---

## Architecture

```text
  Official Siemens PDFs (11 manuals, 3,679 pages)
                    │
              ingestion.py  ── page + equipment_type metadata
                    │
             vectorstore.py ── chunking, embeddings, persistent Chroma
                    │
   ┌────────────────┴─────────────────┐
   │  retrieval                       │
   │   • CodeAwareRetriever  (literal fault-code lookup, no embeddings)
   │   • RerankingRetriever  (wide pool → one LLM reordering call)
   └────────────────┬─────────────────┘
                    │
  ┌─────────────────┴──────────────────────────────────────────┐
  │  agent.py — LangGraph diagnostic agent                     │
  │    tools: search_manuals · get_maintenance_history         │
  │    context: selected machine · photo classification        │
  │             · operator-written past resolutions            │
  └─────────────────┬──────────────────────────────────────────┘
                    │
             safety.py ── live gate: rewrite / block / off
                    │
    ┌───────────────┼───────────────┬──────────────┐
    │               │               │              │
 cost.py        audit.py        cache.py     recurrence.py
 (meter +      (SQLite trail,  (opt-in       (prior occurrences,
  daily cap)    resolutions,    semantic      zero LLM calls)
                CMMS export)    cache)
    └───────────────┴───────────────┴──────────────┘
                    │
            services.py ── the application seam: one diagnostic turn,
                           no Streamlit, no HTTP
                    │
        ┌───────────┴───────────┐
     app.py                  api/main.py
   (Streamlit UI)        (FastAPI + Dockerfile)
```

`services.run_diagnostic()` is the seam everything wraps. The Streamlit app reads widgets and
renders; the API does JSON. Neither owns any pipeline logic.

---

## Engineering decisions worth reading

Each of these came out of a measured failure, not a design document.

### Fault codes are looked up literally, not searched semantically

Embeddings represent *meaning*, and `F30021` has none — so a bare-code query landed on whichever
fault page happened to be semantically nearest, and with no relevance floor the retriever always
returned five confident neighbours. An unknown code produced real, well-cited manual text about a
*different fault*. Codes now go through Chroma's full-text filter (`where_document $contains`) via
`fault_codes.py` + `CodeAwareRetriever`; semantic search handles everything else, unchanged.

| Querying with the code alone (17 verified codes) | Contains the code | Brings the definition page |
|---|---|---|
| Before — semantic + reranker | 14/17 (82%) | **3/17 (18%)** |
| After — code-aware lookup | 17/17 (100%) | **17/17 (100%)** |

The guardrail matters more than the accuracy: zero literal hits is an unambiguous signal, so
`F99999` now returns *"not documented in the available manuals"* instead of describing another
fault. A typo like `F3OO21` (letter O for zero) normalises onto a real code and the app **asks**
which was meant rather than silently correcting.

### The safety gate was silently disabled in 4 of the 5 languages

The gate skips the LLM judge when an answer recommends no physical action — a cheap shortcut that
used a regex of **English** action verbs. So a Portuguese, French, Spanish or German answer never
matched, never reached the judge, and was never gated. The gate was effectively off for most of
the app's users. Fixed: the shortcut applies only to English; everything else always goes to the
judge. This was found by chasing a user-reported symptom ("why is the Portuguese answer nothing
like the English one?"), not by reading the code.

### The prior-occurrence panel reports, and refuses to recommend

The obvious feature is "rank past fixes and suggest the most common one". On the real data that
would be actively dangerous: VFD-04's `F30805` was answered twice with a power-unit module
replacement (and came back), then three times with a power cycle — a frequency ranking would
recommend masking a recurring hardware fault 3-to-2. `recurrence.py` therefore counts, sorts and
dates, and lets the human decide. It runs with **zero LLM calls**, so it costs nothing and cannot
hallucinate.

### The reranker buys ordering, not just coverage

Measured over 37 scenarios (`notebooks/11_retrieval_benchmark.ipynb`): the reranker reaches the
same hit-rate as plain `k=10` (86.5%) with a clearly better MRR (0.685 vs 0.619) while returning
**half the context**.

| Axis | Result |
|---|---|
| **A — chunk size** | 400 → 64.9% / MRR 0.433 · **800 (current) → 78.4% / 0.596** · 1600 → **81.1% / 0.610** |
| **B — search strategy** | **similarity 81.1% / 0.610** · mmr 73.0% / 0.591 · threshold — *invalid, reported as broken* |
| **C — reranker** | OFF 81.1% / 0.610 · **ON 86.5% / 0.685** (+5.4pp, 8s → 45s) |
| **D — k** | k=3 75.7% · **k=5 81.1%** · k=10 86.5% (hit-rate rises mechanically; MRR barely moves) |
| **Combined vs current** | 800/sim/k=5 → 78.4% / 0.596 · **1600/sim/k=5/rerank → 86.5% / 0.685** |

At 400 characters, `vfd_fault_code` accuracy collapses from 82% to 59% — a fault entry's *Cause*
and *Remedy* end up in different chunks. The `similarity_score_threshold` arm is reported as an
**invalid measurement** rather than a bad one: the collection uses `hnsw.space='l2'`, so
LangChain's Euclidean relevance function returned scores outside 0–1 and the threshold discarded
chunks regardless of relevance. Full write-up: `Retrieval_Benchmark_Report.pdf`.

### Results reported as measured, including the unflattering ones

Agent vs. single-chain RAG baseline over 37 verified scenarios
(`notebooks/10_evaluation_baseline.ipynb`):

| | Agent | Baseline |
|---|---|---|
| Root-cause accuracy | **78.4%** | 75.7% |
| Evidence accuracy | 45.9% | **48.6%** |
| Machine-history awareness | cites the real event | structurally 0% — no history tool |
| Conflicting signals (photo says damage, text says "cosmetic scuff") | asks a clarifying question | *"likely safe to continue running the drive"* |
| Safety-precaution ordering failures | 12/30 answers (40.0%) | 32/34 (94.1%) |

The agent does **not** win on evidence accuracy, and a prompt rule alone leaves a 40% ordering
failure rate — which is exactly why the post-hoc safety audit later became a live blocking gate.
Vision (frozen ResNet18 features + logistic regression over a curated 4-category MVTec AD subset):
**82.2%** accuracy vs. a 77.1% majority-class and a 42.5% zero-shot-LLM baseline — the zero-shot
model *underperforms the trivial baseline*, written up rather than hidden.

Two features were built, measured, and **deliberately reverted**: a real-thermal-image vision
dataset (369 images turned out to be only 11 independent scenes, so no split could measure
generalisation) and a component-identity classifier (100% accurate, but each category had its own
capture background, so transfer to a real photo was never established). Both are in
[`docs/limitations_and_opportunities.md`](docs/limitations_and_opportunities.md).

---

## Production concerns, built in

Everything here is **opt-in and off by default** — a fresh clone with only `OPENAI_API_KEY`
behaves exactly like the original RAG app.

| Concern | How it is handled |
|---|---|
| **Configuration** | Typed `Settings` dataclass (`config.py`), every knob a `FACTORY_FLOOR_*` env var with a safe default |
| **Secrets** | `secrets.py` is a seam — `env` today, with `aws` / `vault` / `doppler` / `sops` designed in [`docs/secrets.md`](docs/secrets.md). The key never appears in a `repr` |
| **Cost** | Per-token metering, per-session total, a daily ledger and a hard spend cap — including the safety gate's own LLM calls |
| **Safety** | Live gate: `off` \| `rewrite` \| `block`, applied before the operator sees the answer |
| **Auditability** | SQLite (WAL) trail of every recommendation, its sources, tools, cost and operator; operator-written resolutions; a CMMS-export demo |
| **Identity** | `operators.csv` with PBKDF2-hashed PINs, managed via `manage_operators.py` (never hand-edited) |
| **Caching** | Opt-in semantic answer cache (Chroma, cosine), exact-match for fault codes |
| **Multi-tenancy** | Collection-per-tenant seam in `tenancy.py`; the full design in [`docs/multi_tenancy.md`](docs/multi_tenancy.md) |
| **Serving** | FastAPI proof (`/health`, `POST /diagnose`) + Dockerfile; the full surface designed in [`docs/backend_architecture.md`](docs/backend_architecture.md) |
| **Tests / CI** | 183 tests (unit + integration, no API key and no cost), ruff, GitHub Actions on Python 3.12 and 3.13 |
| **Tracing** | LangSmith project routing + permalink per run (`tracing.py`) |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                 # add your OPENAI_API_KEY
```

Then build the corpus and the vector store, once:

```bash
python download_manuals.py          # 11 official Siemens PDFs → data/manuals/
python download_defect_images.py    # MVTec AD subset → data/defect_images/  (vision only)
```

Run `notebooks/01_data_ingestion.ipynb` and `notebooks/02_vector_database.ipynb` to create the
persistent Chroma store under `data/vectorstore/`. Then:

```bash
make app          # streamlit run app.py
make test         # 183 tests, no API key needed, no cost
make lint         # ruff
```

`make` targets: `test` · `test-all` · `test-llm` · `lint` · `fmt` · `nb` (re-execute every
notebook as a regression check) · `app`. Override the interpreter with
`make test PYTHON=/path/to/python`.

**As an API instead of a UI:**

```bash
uvicorn api.main:app --reload        # /health, POST /diagnose
docker build -t factory-floor . && docker run -p 8000:8000 -v "$PWD/data:/app/data" factory-floor
```

**Notebooks 01–14** walk the whole build in order: ingestion, vector DB, RAG, conversation
history, maintenance history, computer vision, the agent, tracing, the safety validator,
evaluation, the retrieval ablation, the service layer, the audit trail, and the semantic cache.
Notebooks 01–02 rebuild `data/vectorstore/`; the rest run against it.

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

**Corpus — 11 official Siemens manuals, 3,679 pages**, balanced across both equipment families
and five document types (operating instructions, list/fault manuals, function manuals,
engineering manual, catalog): SINAMICS G120 / G120C / CU240B-2 / CU240E-2 on the drive side,
SIMOTICS GP / SD / DP / 1LE1 / 1LE7 on the motor side. ~2,766 pages VFD / ~913 pages motor.
Every row of `manual_sources.csv` carries an explicit `equipment_type` that ingestion reads
rather than guessing from the filename. Sources are official Siemens links; if one breaks, the
URL is in the manifest for a manual download.

**Fleet — 20 simulated assets** (10 motors + 10 VFDs) in `machines.csv`, with 2–6
fault/repair/preventive events each in `maintenance_history.csv`, generated once with a fixed
seed and committed as static data.

**Fault codes — 368 real codes** in `fault_codes.csv`, generated by `build_fault_code_index.py`
from the raw PDF text (a full regex scan of every page, 191 real F-codes found in the list
manuals) — not invented, and not sampled from a semantic search. An earlier draft used
plausible-looking codes like `F0051` that simply do not exist in the manuals and returned nothing.

<details>
<summary><strong>The 17 verified codes used across the maintenance history and the evaluation set</strong></summary>

| Code | Meaning |
|---|---|
| F30001 | Power unit: Overcurrent |
| F30002 | Power unit: DC link voltage overvoltage |
| F30003 | Power unit: DC link voltage undervoltage |
| F30004 | Power unit: Overtemperature heat sink AC inverter |
| F30011 | Power unit: Line phase failure in main circuit |
| F30021 | Power unit: Ground fault |
| F30035 | Power unit: Air intake overtemperature |
| F30037 | Power unit: Rectifier overtemperature |
| F30059 | Power unit: Internal fan faulty |
| F30805 | Power unit: EEPROM checksum error |
| F30950 | Power unit: Internal software error |
| F07011 | Drive: Motor overtemperature |
| F07016 | Drive: Motor temperature sensor fault |
| F07801 | Drive: Motor overcurrent |
| F07807 | Drive: Short-circuit/ground fault at motor-side output terminals |
| F07860 | External fault 1 |
| F07901 | Drive: Motor overspeed |

</details>

**Electric motors have no numeric fault-code system at all** — confirmed by scanning all five
motor manuals with the same regex: zero matches. Motor events use the manuals' own terminology
instead (abnormal noise, insulation resistance below the specified value, condensation in the
terminal box, re-greasing per the lubricant plate, direction of rotation with the motor
uncoupled).

---

## Project structure

```text
├── app.py                      # Streamlit UI — a thin view over services.run_diagnostic()
├── api/main.py                 # FastAPI proof: /health, POST /diagnose
├── Dockerfile                  # container for the API (data/ mounted at runtime)
├── Makefile                    # test / lint / fmt / nb / app
├── manage_operators.py         # list | add | setpin | remove (PBKDF2, never hand-edited)
├── build_fault_code_index.py   # regex scan of the PDFs → fault_codes.csv
├── download_manuals.py         # official Siemens PDFs → data/manuals/
├── download_defect_images.py   # MVTec AD subset → data/defect_images/
├── factory_floor/
│   ├── config.py               # typed Settings — the authoritative knob list
│   ├── secrets.py              # get_secret() — the seam a managed vault plugs into
│   ├── identity.py             # operator sign-in (PBKDF2 PIN hashes)
│   ├── tenancy.py              # multi-tenant seam — resolve_collection(tenant_id)
│   ├── ingestion.py            # PDF loading + metadata tagging
│   ├── vectorstore.py          # chunking, embeddings, Chroma build/load
│   ├── rag.py                  # prompt, retrievers, ask(), history-aware reformulation
│   ├── fault_codes.py          # literal code lookup + normalisation of typos
│   ├── machines.py             # machine registry + per-machine maintenance history
│   ├── recurrence.py           # prior occurrences of a fault — no LLM, reports only
│   ├── manuals.py              # per-page PDF extraction for "download this page"
│   ├── agent.py                # the diagnostic agent — tools, safety-first contract
│   ├── services.py             # THE SEAM: one diagnostic turn, no UI, no HTTP
│   ├── safety.py               # post-hoc audit + live blocking gate (enforce_safety)
│   ├── cost.py                 # token metering, session usage, daily ledger + cap
│   ├── cache.py                # opt-in semantic answer cache
│   ├── audit.py                # SQLite trail + writable resolutions + CMMS demo
│   ├── vision.py               # defect classification (trained + zero-shot)
│   ├── defect_dataset.py       # MVTec AD subset scan + stratified manifest
│   ├── tracing.py              # LangSmith project routing + run permalinks
│   ├── evaluation.py           # eval_scenarios.csv loader, scoring, agent-vs-baseline
│   └── retrieval_benchmark.py  # the ablation-study harness
├── tests/                      # 183 tests — unit + integration, no API key, no cost
├── notebooks/                  # 01–14, the build in order
├── docs/
│   ├── project_log.md          # the full dated build history
│   ├── backend_architecture.md # the intended full API surface
│   ├── multi_tenancy.md        # the multi-tenant design
│   └── secrets.md              # vault backends, designed not built
├── machines.csv · maintenance_history.csv · operators.csv · tenants.csv
├── manual_sources.csv · fault_codes.csv · defect_image_manifest.csv
├── eval_scenarios.csv          # 37 verified troubleshooting scenarios
└── data/                       # gitignored: manuals, vectorstores, images, audit DB
```

---

## Limitations

- **Not a certified safety system.** It is decision support for a trained technician. The safety
  gate reduces precaution-ordering failures; it does not eliminate them, and the measured
  residual failure rate is published above rather than tuned until it looked good.
- **The fleet and its maintenance history are simulated.** The manuals, the fault codes and the
  measured retrieval/evaluation numbers are real.
- **Vision runs on MVTec AD**, an industrial-defect dataset that is not photographs of motors.
  Transfer to a real photo taken inside a drive cabinet is explicitly not established.
- **Evidence accuracy (45.9%) is the weakest number in the project** and is honestly the next
  thing worth fixing — hybrid/lexical retrieval is the open lead.
- **Designed but not built:** a real vault backend, live multi-tenant deployment, the full API
  surface with streaming, compose + load balancing. Each has a design doc and a seam, not an
  implementation.

Open gaps and abandoned experiments are tracked in
[`docs/limitations_and_opportunities.md`](docs/limitations_and_opportunities.md); the dated build history is
in [`docs/project_log.md`](docs/project_log.md).

---

## Background

Originally the final project for the Ironhack AI Engineering bootcamp — all seven core
requirements (RAG with citations, agents, tracing, a multimodal component, a ≥30-task evaluation
set with baseline benchmarking, web deployment, presentation) were met, and the compliance
checklist is preserved in [`docs/project_log.md`](docs/project_log.md). Everything above the
bootcamp line — the service layer, cost control, the live safety gate, the audit trail, identity,
caching, tenancy seams, the API and the test suite — was built afterwards to take it from a
working demo to something that could survive contact with a real deployment.

## License

MIT — see [LICENSE](LICENSE).
