# The Factory Floor

Industrial Maintenance Copilot for **electric motors + variable-frequency drives (VFDs)**.

This package implements the project up to the following build order:

1. Choose equipment family ✅
2. Collect PDFs ✅ (official source manifest + downloader)
3. Extract documents ✅
4. Chunking + embeddings ✅
5. Vector DB ✅
6. Retriever ✅
7. RAG with citations ✅
8. Basic Streamlit interface ✅
9. Conversation history ✅ (history-aware follow-up questions)
10. Maintenance history ✅ (owner-confirmed done 2026-08-19 — simulated structured fault/repair log per machine, machine selector + history display + RAG filter working)

## What changed on 2026-08-25

Six commits, in the order they happened. Detail for each is further down; this is the map.

| # | Change | Headline result |
|---|---|---|
| 1 | **Reranker** (`rag.py::rerank_documents`, `RerankingRetriever`) | Fetches a wider pool and reorders it with one LLM call. Later measured: same coverage as `k=10` with better ordering on half the context |
| 2 | **Photo-context memory fix** (`rag.py::build_chat_history`) | A follow-up after a photo no longer forgets the classification and re-asks for the image |
| 3 | **Machine context for the agent** (`agent.py::format_machine_context`) | The agent is told which machine is selected, so it stops asking whether it is a motor or a VFD — information the sidebar already established |
| 4 | **Reranker output leak fixed** (`agent.py::stream_diagnostic_agent`) | The reranker's ranking digits (`0,14,13,...`) were streaming to the operator ahead of the answer; filtered by `langgraph_node` |
| 5 | **Exact fault-code lookup** (`fault_codes.py`, `CodeAwareRetriever`) | Bare-code queries: definition page **18% → 100%**. Unknown codes now refused instead of answered from a similar one |
| 6 | **Retrieval ablation study** (`notebooks/11_retrieval_benchmark.ipynb`) | chunk size / search strategy / reranker / k, measured. Combined best config **+8.1pp hit-rate, +0.088 MRR** over current |

Two things were tried and **deliberately abandoned**, both written up rather than quietly dropped:

- **Replacing the MVTec vision dataset with real thermal motor images.** The candidate dataset
  trains to 98.9% but its 369 images are only **11 independent scenes** of near-identical frames
  (neighbouring frames differ by 0.55–2.47 grey levels out of 255), with one scene for three of
  the four classes — so no split can measure generalisation. Every dataset checked and the cheap
  neighbouring-frame test that exposed it are in `dificuldades_e_oportunidades.md` #12.
- **A component-identity classifier** on the existing MVTec data. It worked (100% vs a 31.4%
  majority baseline, robust to greyscale/inversion/rotation/noise) and did fix the agent's
  behaviour, but was reverted at the owner's call: the categories each have their own capture
  background, so transfer to a photo taken inside a real drive was never established.

New artefacts: `fault_codes.csv` (368 real codes), `Retrieval_Benchmark_Report.pdf` (the ablation
study as a standalone 4-page report), `build_fault_code_index.py`, and roadmap slide 5.

## Compliance checklist (bootcamp core requirements) — added 2026-08-19

This project is the final project for an AI Engineering Bootcamp ("The Factory Floor",
project 9 of an 11-project pack). The official spec defines 7 non-negotiable core
requirements, separate from the numbered build order above (which tracks the RAG
pipeline specifically). Cross-checking this project against those 7 requirements on
2026-08-19 surfaced real gaps — reported honestly here rather than left implicit:

| # | Requirement (official spec) | Status | Notes |
|---|---|---|---|
| 1 | Retrieval-Augmented Generation, cited sources | ✅ Done | Chroma vector DB, citations in every answer (`factory_floor/rag.py`) |
| 2 | Agents — model decides which tools to use, carries memory | ✅ Done | `factory_floor/agent.py` — real agentic architecture (`langchain.agents.create_agent`, LangGraph under the hood) with `search_manuals` and `get_maintenance_history` tools; the model decides which to call, combines them with any photo classification, and asks a clarifying question instead of guessing when signals conflict (verified with a real conflicting-signals test in `notebooks/07_orchestrator_agent.ipynb`) |
| 3 | Tracing / observability (LangSmith, Langfuse, or Arize Phoenix) | ✅ Done | `factory_floor/tracing.py` routes every run to a dedicated `factory-floor` LangSmith project (without editing the shared `.env`, which pins an unrelated project). Every `run_diagnostic_agent()`/`ask()` call is tagged, named, and returns a `run_id`; `factory_floor/tracing.py::run_url()` builds a permalink for either. `notebooks/08_tracing_observability.ipynb` verifies a real multi-step agent trace (2 different tools, in order), a single-chain baseline trace, and a zero-tool-call reasoning trace, all in the same project |
| 4 | A multimodal component | ✅ Done | Defect classification over a curated MVTec AD subset (`cable`, `metal_nut`, `screw`, `transistor`) — `factory_floor/vision.py`, `notebooks/06_computer_vision.ipynb`, trained classifier 82.2% accuracy vs. 77.1% majority-class / 42.5% zero-shot-LLM baselines on a held-out test split |
| 5 | Evaluation set (≥30 tasks) + benchmarking against a baseline | ✅ Done | 37 troubleshooting scenarios (`eval_scenarios.csv`), each with a verified expected root cause + manual evidence phrase, scored via `factory_floor/evaluation.py` and benchmarked agent-vs-baseline (`rag.ask()`) in `notebooks/10_evaluation_baseline.ipynb`: root-cause accuracy 78.4% (agent) vs. 75.7% (baseline), evidence accuracy 45.9% vs. 48.6%, plus 3 structural comparisons — machine-history awareness (agent correctly cites past events, baseline structurally cannot), conflict handling (agent asks a clarifying question on a real held-out conflicting-signals case, baseline confidently reassured that it's "likely safe to continue running"), and the safety audit below. Vision (item 4) supplies the required defect-classification accuracy part of this same requirement |
| 6 | Deployment as a web application | ✅ Done | Streamlit app, runs locally (`app.py`) |
| 7 | Final presentation (15–20 min) | — | Delivered live at the end of the 5-week timeline, not applicable to the repo itself |

**Correction note on item 05 of `roadmap.pptx` ("Computer Vision"):** the roadmap's original
description ("the user uploads a photo of a nameplate or fault display, the system reads
that image and extracts the fault code/model") does **not** match the official spec's
multimodal-core requirement, which is defect-*type* classification (scratch, crack,
contamination...) over the MVTec Anomaly Detection dataset, with a required
classification-accuracy metric on a held-out test set. This was caught by re-reading the
official project requirements documents mid-project, not assumed from the start. The
roadmap.pptx now has two additional slides (appended after the original two, which are
untouched) documenting this checklist and the corrected item 05 description. See
`CLAUDE.md`'s 2026-08-19 note for the full reasoning. As of 2026-08-21, all 7 core
requirements in the table above are ✅ Done — see `dificuldades_e_oportunidades.md` for the
non-core gaps that remain (SDS corpus, hybrid/lexical retrieval for exact fault codes,
a live-blocking Safety Validator). The project is tracked in a real Git repository
(`github.com/mcorreia10/Factory_Floor_Chatbot`), confirmed 2026-08-24 — no longer an
open gap.

## Retrieval benchmark — ablation study (`notebooks/11_retrieval_benchmark.ipynb`)

Notebook 10 asks *"is my system better than the obvious alternative?"*. Notebook 11 asks the
other question — *"why this configuration and not another?"* — varying one design axis at a
time over the same 37 scenarios. Scored with **zero LLM calls** (except the reranker arm):
every scenario's `expected_evidence_keywords` phrase was verified to exist in the target
chunk, so a phrase match is a real chunk-level hit test.

`code_aware` is off throughout, deliberately — with it on, the 17 fault-code scenarios would
score perfectly under every configuration and flatten the differences being measured.

| Axis | Result |
|---|---|
| **A — chunk size** | 400 → 64.9% / MRR 0.433 · **800 (current) → 78.4% / 0.596** · 1600 → **81.1% / 0.610** |
| **B — search strategy** | **similarity 81.1% / 0.610** · mmr 73.0% / 0.591 · threshold — *invalid, see below* |
| **C — reranker** | OFF 81.1% / 0.610 · **ON 86.5% / 0.685** (+5.4pp, +0.075 MRR, 8s → 45s) |
| **D — k** | k=3 75.7% · **k=5 81.1%** · k=10 86.5% (hit-rate rises mechanically; MRR barely moves) |
| **Combined vs current** | 800/sim/k=5 → 78.4% / 0.596 · **1600/sim/k=5/rerank → 86.5% / 0.685** (+8.1pp, +0.088) |

Three findings worth the space:

- **Small chunks hurt, and the category breakdown says why**: at 400 characters
  `vfd_fault_code` collapses from 82% to 59%, because a fault entry's *Cause* and *Remedy*
  end up in different chunks.
- **The reranker earns its keep, and not just by fetching more.** It reaches exactly the same
  hit-rate as plain `k=10` (86.5%) but with a clearly better MRR (0.685 vs 0.619) while
  returning **half the context** — so the LLM call is buying ordering, not merely coverage.
- **The `similarity_score_threshold` arm is invalid, not merely worse.** The collection uses
  `hnsw.space='l2'`, so LangChain's Euclidean relevance function returned scores outside 0-1
  (several negative) and the threshold discarded chunks regardless of relevance. Reported as
  a broken metric rather than quietly dropped; testing it properly needs a cosine-space
  collection.

Production config was **not** changed on the strength of this: moving to `chunk_size=1600`
means rebuilding the main store and re-running notebooks 02-10, which is a separate decision.

The same study is written up as a standalone 4-page report in
**`Retrieval_Benchmark_Report.pdf`** — method, all four axes, the combined-vs-current
comparison and the limitations, in a form that can be handed to someone who will not open a
notebook.

The variant stores are gitignored and rebuildable in ~3 min each. To reclaim the ~434 MB:
`rm -rf data/vectorstore_cs400 data/vectorstore_cs1600`

## Session note (2026-08-25) — read this FIRST before continuing

Most recent session. Supersedes the notes below for "what to do next".

**Bare fault codes now work.** An operator typing just `F30021`, with no symptom, was the
project's oldest and most dangerous weakness (difficulty #1): embeddings represent meaning, and a
code has none, so a bare-code query landed on whichever fault page was semantically nearest.
Worse, with no relevance floor the retriever always returned five neighbours, so an unknown code
still produced real manual text about *a different fault*, well cited and confidently explained.

Codes are now **looked up literally, not searched semantically** — via Chroma's own full-text
filter (`where_document={"$contains": ...}`, trigram index), so no new dependency and no
re-indexing. New `factory_floor/fault_codes.py` + `CodeAwareRetriever` in `factory_floor/rag.py`;
queries with no code delegate to the existing semantic path completely unchanged.

Measured on the 17 verified codes, querying with **the code alone**:

| | Contains the code | Brings the definition page |
|---|---|---|
| Before (semantic + reranker) | 14/17 (82%) | **3/17 (18%)** |
| After (code-aware) | 17/17 (100%) | **17/17 (100%)** |

**The guardrail matters more than the accuracy.** Zero literal hits is a clean, unambiguous
signal, so an unknown code now yields an explicit not-found notice plus rule 1b of
`DIAGNOSTIC_SYSTEM_PROMPT`. Verified live: `F99999` alone returns *"not documented in the
available manuals … please re-check the fault code on the equipment display"* instead of
describing a different fault.

**Typos are handled separately from unknown codes.** `F3OO21` (letter O for zero) normalises onto
a real code and the app **asks** which was meant, rather than silently correcting. New
`fault_codes.csv` (368 real codes, generated by `build_fault_code_index.py`) backs this and is
committed as static reference data, same convention as `machines.csv`.

## Session note (2026-08-21) — read this FIRST before continuing

Most recent session. Closed the last 2 of the 7 core requirements (Tracing, Evaluation) plus the
Safety Validator (roadmap item 06) — all 7 core requirements and all 9 roadmap items are now
✅ Done. Supersedes the 2026-08-19 note below for "what to do next" — that note is kept for
history/context on the Orchestrator Agent milestone, but the resume point is this one.

**Tracing (core requirement 3)**: `factory_floor/tracing.py` — `configure_tracing()` routes every
run to a dedicated `factory-floor` LangSmith project without editing the shared `.env` (which pins
an unrelated project, `lca-lc-foundation`, reused across other bootcamp labs). Real gotcha found and
handled: `langsmith` 0.11.0's `get_env_var()`/`get_tracer_project()` are `functools.lru_cache`'d, so
the override calls `.cache_clear()` on both — otherwise a later `os.environ` change is silently
ignored. `run_diagnostic_agent()` and `ask()` both tag/name every call and return a `run_id`;
`tracing.py::run_url()` builds a permalink for either (needed because `LangChainTracer.get_run_url()`
only sees callback-based runs and raises `"No traced run found"` for `ask()`'s `@traceable`-decorated
calls — `Client.read_run()` was avoided entirely, it's deprecated and needs `project_id`+`start_time`
on this backend). Verified live in `notebooks/08_tracing_observability.ipynb`: a real multi-step
agent trace (the agent chose to call `get_maintenance_history` then `search_manuals`, in that order,
for a real F30059 fan-fault question on VFD-06), a single-chain baseline trace for contrast, and the
zero-tool-call conflicting-signals reasoning trace from the Orchestrator Agent milestone.

**Safety Validator (roadmap item 06)**: scope, confirmed with the owner before building — an
**audit utility used in the evaluation notebook**, not a live blocking/warning gate in `app.py` (the
original architecture diagram's fuller real-time version remains a possible future step). Added rule
6 to `factory_floor/agent.py::DIAGNOSTIC_SYSTEM_PROMPT` — outranks every other rule, requires a
"Safety precautions" section before any physically-performed step, grounded in the manuals' own
safety-instruction sections (no new SDS corpus needed for this). New `factory_floor/safety.py` —
`check_safety_precautions()` (LLM-as-judge, mirrors `vision.py`'s structured-output pattern),
`check_safety_precautions_keyword()` (deterministic regex cross-check, guards against the judge's own
self-preference bias), `audit_answers()` (aggregate). `notebooks/09_safety_validator.ipynb` verifies
the judge against 3 hand-written probes and runs a real A/B (same question, same agent, rule 6
present vs. temporarily stripped via monkeypatch) — the contracted version passes, the uncontracted
one fails, on real API calls.

**Evaluation with baseline benchmarking (core requirement 5)**: new `eval_scenarios.csv` — 37
troubleshooting scenarios (≥30 required), each with an `expected_root_cause_keywords` and
`expected_evidence_keywords` individually verified against the real vectorstore (the evidence phrase
is always drawn from the manual's actual Cause/Remedy text and confirmed absent from the question
itself, so a model can't score a hit by parroting). New `factory_floor/evaluation.py` scores both
`rag.ask()` (baseline) and `run_diagnostic_agent()` (agent) identically. Real measured results,
`notebooks/10_evaluation_baseline.ipynb`:
- Root-cause accuracy: 78.4% agent vs. 75.7% baseline. Evidence accuracy: 45.9% vs. 48.6% (close —
  an honest echo of difficulty #1, embedding retrieval is measurably unreliable for exact fault codes
  without extra machinery this project doesn't have yet).
- **Machine-history awareness**: baseline structurally scores 0% (asserted, not just observed — it
  has no history tool at all); the agent correctly cited the real 2022-06-25 history date.
- **Conflict handling**: on the held-out `structural_damage` photo + "just a tiny cosmetic scuff"
  text, the agent asked a clarifying question; the baseline — which has no `vision_context` mechanism
  at all — took the text at face value and concluded "it is likely safe to continue running the
  drive," a real, slightly concerning illustration of the gap between the two designs.
- **Safety audit** (Part 3, run over all 37 real answers from both pipelines): the agent (rule 6
  active) had **12 precaution-ordering failures out of 30 answers that recommended an action**
  (40.0% failure rate); the baseline (deliberately uncontracted) had **32 failures out of 34**
  (94.1%). Judge/keyword agreement 91.9% (agent) / 78.4% (baseline). Reported as measured — the rule
  substantially reduces the failure rate but does not eliminate it, confirmed independently by a live
  Streamlit UI test the same session (a real answer shown to an operator put the safety section after
  the repair steps despite rule 6 being active). This is a genuine LLM instruction-following
  reliability limit, not a bug in the prompt — flagged honestly rather than tuned until it looked
  better, per the spec's own wording ("reporting it honestly is the point").
- Vision (item 4) supplies Part 1 of this same requirement, referenced not re-run: 82.2% trained vs.
  77.1% majority-class vs. 42.5% zero-shot baselines.

**All 7 core requirements are now ✅ Done.** What remains, not core, tracked in
`dificuldades_e_oportunidades.md`: the SDS corpus (item 11), hybrid/lexical retrieval or a
post-retrieval fault-code guardrail for the modest evidence-accuracy numbers above (item 1), a
live-blocking version of the Safety Validator, and the 3-page project document (deliverables
outside this session's agreed scope). The real Git repository is done (confirmed 2026-08-24) —
see the note below.

## Session note (2026-08-19) — read this before continuing

Supersedes the 2026-08-18 note below for "what to do next" on the Maintenance History and
Orchestrator Agent milestones — that note is kept for history/context, but the resume point above
(2026-08-21) is the current one.

**What got done today:**
- **Caught a real scope bug**: the internal `roadmap.pptx`/README description of item 05
  ("Computer Vision" = read a nameplate/fault-code photo) did not match the official bootcamp
  requirements doc (`Downloads/Requirements/09_The_Factory_Floor.pdf`), which requires classifying
  the *type of defect* visible in a photo (scratch/crack/contamination...) over the MVTec Anomaly
  Detection dataset, with a required accuracy metric. Corrected everywhere (see Compliance
  checklist above).
- **Built the full Vision path end to end, with real data and real results**: 4 curated MVTec AD
  categories physically plausible on/around motors and VFDs (`cable`, `metal_nut`, `screw`,
  `transistor` — not the 15 generic categories) — 1502 real images, downloaded via the
  `Voxel51/mvtec-ad` Hugging Face mirror (`download_defect_images.py`) since the official mvtec.com
  archive is gated/broken. New `factory_floor/defect_dataset.py` (dataset scan + stratified
  manifest) and `factory_floor/vision.py` (`classify_defect_trained`, `classify_defect_zero_shot`,
  `recommend_actions`, `evaluate_classifier`). `notebooks/06_computer_vision.ipynb` executed clean.
  **Real measured results**: trained classifier (frozen ResNet18 features + LogisticRegression)
  82.2% accuracy vs. 77.1% majority-class baseline vs. 42.5% zero-shot-LLM baseline (the zero-shot
  model actually underperforms the trivial baseline — a genuine failure-analysis finding, see
  `dificuldades_e_oportunidades.md` #13).
- **`app.py` reshaped around live operator feedback, twice**: (1) added `recommend_actions()` so a
  photo doesn't just get a bare label — the operator gets concrete next steps with safety
  precautions first, explicitly flagged as *not* manual-grounded; (2) unified photo submissions
  into the same `st.session_state["turns"]` conversation list text questions use, so follow-up
  questions work after a photo too (turns are tagged `"type": "photo"` vs `"type": "text"`); (3)
  merged the photo upload into the main question form as an "OR" alternative with a single submit
  button, matching an owner-supplied mockup; (4) removed the probability-chart/zero-shot-compare
  detail from the operator-facing UI (still available in the notebook, not useful to an operator).
  Full reasoning trail in `CLAUDE.md`'s 2026-08-19 notes.
- **`roadmap.pptx` updated in place**: item 04 (Maintenance History) and item 05 (Computer Vision)
  marked `Done` (owner-confirmed explicitly, not assumed), progress counter `44%/4 of 9` →
  `67%/6 of 9`. Two new slides appended at the end (checklist of the 7 core requirements, and the
  item-05 scope-correction explanation) — the original 2 slides were never overwritten, only added
  to. Known gap: only status *text* was changed, not each card's colored background tint/badge —
  couldn't verify the shape-to-color mapping safely without rendering the slide (no LibreOffice in
  this environment); left alone rather than risk recoloring the wrong shape.
- **Explored, then explicitly deferred**: an audio-based diagnostic path (operator records the
  motor/VFD sound, system suggests likely fault + checks) — not required by the bootcamp's 7 core
  requirements (already satisfied by the vision path), and the one genuinely domain-relevant free
  dataset found (UOEMD-VAFCVS, University of Ottawa — real induction motor + real VFD + real fault
  types) is small (128 recordings) and needs real audio-preprocessing work before any model can be
  trained on it. Fully written up as opportunity #4 in `dificuldades_e_oportunidades.md`, including
  the dataset details and why it's harder than the MVTec case — don't re-research this from scratch
  next time, start from that note.

**Resume point, updated same day (2026-08-19, cont.) — Orchestrator Agent built**: the biggest of
the 3 remaining gaps is done. New `factory_floor/agent.py` — `build_rag_tool()` (thin retrieval,
`retriever.invoke()` + `rag.format_context()`, no second LLM synthesis inside the tool — the agent
itself does the final citing/combining), `build_history_tool()` (machine_id closed over, never
LLM-controlled — it's a fact of the request context from the sidebar, not something the model should
guess), `build_diagnostic_agent()` and `run_diagnostic_agent()` (single entry point, same shape as
`rag.ask()`). Built on `langchain.agents.create_agent` (the non-deprecated successor to
`langgraph.prebuilt.create_react_agent` in this environment's LangChain 1.x — confirmed via a live
deprecation warning during development, not assumed). A photo is still classified *before* the agent
reasons (not exposed as an LLM-callable tool — there's no real "whether to classify" decision once a
photo is uploaded); the classification result is injected as text context, and the agent decides
what to do with it (search manuals about that defect type, check history, or answer directly).
`app.py`'s `submit_turn()` now routes every question — text, photo, or both together — through this
agent; the form no longer forces a choice between the two. Each answered turn shows a "Tools used"
trace in the UI. `recommend_actions()` is retired from the main flow (owner-confirmed) — the agent
now produces the final recommendation itself, informed by whichever tools it chose.

**Verified, not just built**: 4 real scenarios run directly against `run_diagnostic_agent()`, plus
the same 3 core ones re-run inside `notebooks/07_orchestrator_agent.ipynb` with `assert` checks —
including the central one: a real held-out photo classified as `structural_damage` paired with a
text description that downplays it as "just a cosmetic scuff" — the agent asked a clarifying
question instead of picking a side, confirmed programmatically (`assert '?' in answer`). Also
confirmed the history tool is never even offered for a `GENERAL` (no-machine-selected) question.

**Resume point superseded 2026-08-21** — see the session note below (after the 2026-08-18 note)
for how the remaining 2 core requirements (Tracing, Evaluation) were closed, plus the Safety
Validator (roadmap item 06). All 7 core requirements are now ✅ Done — see the Compliance
checklist above and `dificuldades_e_oportunidades.md` for the non-core gaps that remain.

## Session note (2026-08-18) — read this before continuing

For whoever picks this up next, including a fresh AI session: this session built the
Maintenance History milestone (point 10 above). New files: `machines.csv`,
`maintenance_history.csv`, `factory_floor/machines.py`,
`notebooks/05_maintenance_history.ipynb`. `app.py` got a sidebar machine/VFD selector
that (a) filters the RAG search to that machine's `equipment_type` and (b) shows that
machine's maintenance history right after "Show retrieved evidence" for each answered
turn. A "🌐 General question (search all manuals)" option was added as the default so
unfiltered questions still work. The "Example question" shortcut dropdown was removed
on request — operators must type their own question now.

A real bug was caught and fixed along the way: the VFD fault codes originally used in
the simulated maintenance history (F0001, F0003, F0051...) were **invented**, not
real — they don't exist in the ingested manuals, so asking about them returned no
answer. Replaced with 8 verified real Siemens SINAMICS codes extracted directly from
the corpus (see the "Machines & maintenance history" section below for the list, and
`CLAUDE.md`'s dated notes for the full investigation).

**The project owner explicitly does not consider this milestone finished yet** —
that's why the roadmap PPTX (`roadmap.pptx`, also in this folder) marks item 04 as
"On Going", not "Done". Read `CLAUDE.md` in full before making further changes or
declaring anything else done; it has the reasoning and constraints this file doesn't
have room for.

### Next step — done (2026-08-18-d)

The full audit assigned above is complete:

- **VFDs**: scanned the raw text of every page of both List Manual PDFs with a
  regex for the real `F\d{5}`/`A\d{5}` code pattern (not semantic search, which
  proved unreliable for this) — 191 real F-codes found in total. Picked a diverse
  17-code subset (see the table above) covering overcurrent, overvoltage,
  undervoltage, ground fault, three flavors of overtemperature, fan/EEPROM/software
  faults, motor-side overcurrent/overtemperature/overspeed, and an external fault —
  and verified every one of the 17 individually against the vector store
  (`vs.similarity_search` + exact-string match) and through a full `ask()` call.
- **Electric motors**: scanned all 5 motor manuals with the same regex — **zero
  matches**, confirming motors genuinely have no numeric fault-code system. Extracted
  the real terminology they do use instead (abnormal/unusual noise, insulation
  resistance below the specified value, condensation, re-greasing per the lubricant
  plate, direction of rotation with motor uncoupled) and rewrote
  `maintenance_history.csv`'s motor descriptions to use it.
- `maintenance_history.csv` regenerated with this expanded vocabulary (17 VFD codes /
  10 motor issue types), `test_questions.txt` and this README's code table updated to
  match, `notebooks/05_maintenance_history.ipynb` re-executed. Full method and findings
  in `CLAUDE.md`'s 2026-08-18-d note.

### Later the same day (2026-08-18-e/f) — UI fix + live conversation testing

- **`test_questions.txt` fixed**: the shop-floor scenario questions (41-72) no longer
  name a machine ID in the question text itself (e.g. "VFD-05 tripped with..." →
  "Tripped with..."). The sidebar selector already ties the question to a specific
  `machine_id` — an operator who just picked the machine from a dropdown would never
  re-type its ID in the free-text box. Owner caught this.
- **Bug found and fixed**: the "Maintenance history" expander was rendering once
  *per turn* in a conversation (identical table repeated after every Q&A). Moved out
  of `render_turn()` to render once, after the last turn, right before the follow-up
  question box. Verified with a 5-turn conversation — exactly one history expander at
  the end, not five.
- **Live-tested extensively via browser** (not just code-read): individual VFD
  fault-code questions, motor shop-floor scenarios, "General question" mode (no
  filter, still returns correctly-scoped sources), a 3-turn and a 5-turn conversation
  on the same machine. All passed — context correctly carried across turns via
  history-aware reformulation, `equipment_type` filter held throughout, no console
  errors, "← Back to start" correctly clears turns while keeping the machine
  selection. One cosmetic-only finding, not fixed: right after toggling an expander,
  `st.button` labels can briefly paint invisible (text confirmed present in the DOM
  via `get_page_text`, self-resolves on next repaint) — documented in `CLAUDE.md`,
  don't mistake it for a new regression if seen again.

### Resume point for the next session

Everything above is done and verified working. What's still open, in case tomorrow's
session (or a fresh AI session) needs a starting point:
- Point 10 is still marked "On Going" in `roadmap.pptx` and this README — the owner
  hasn't said what "finished" means yet for this milestone. Ask before marking it
  ✅ Done anywhere.
- The maintenance history is still display-only — not fed into the LLM's
  reasoning/prompt. That's intentionally deferred to the future Orchestrator Agent
  step (07 on the roadmap), not a gap in point 10.
- No automated test suite exists — every verification so far (including the 5-turn
  conversation test) was manual, via live browser driving. Roadmap step 09
  (Evaluation) is where that gets addressed, not started yet.

## Architecture at this milestone

```text
Official motor / VFD PDFs
          ↓
      PDF loader
          ↓
   page metadata
          ↓
       chunks
          ↓
      embeddings
          ↓
  persistent Chroma DB
          ↓
      retriever
          ↓
     RAG prompt
          ↓
        LLM
          ↓
 answer + source/page citations
          ↓
      Streamlit UI
```

## Project structure

```text
factory_floor_milestone/
├── app.py
├── download_manuals.py
├── download_defect_images.py
├── manual_sources.csv
├── machines.csv
├── maintenance_history.csv
├── defect_image_manifest.csv
├── eval_scenarios.csv    # 37 troubleshooting scenarios for notebook 10
├── test_questions.txt    # earlier manual-testing draft, superseded by eval_scenarios.csv for scoring
├── roadmap.pptx          # roadmap slide deck, kept in sync with actual progress
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── CLAUDE.md
├── dificuldades_e_oportunidades.md
├── factory_floor/
│   ├── config.py         # shared paths / collection name + typed Settings object
│   ├── secrets.py        # get_secret() — the seam a managed vault plugs into
│   ├── ingestion.py      # PDF loading + metadata tagging (equipment_type read from manual_sources.csv)
│   ├── vectorstore.py    # chunking, embeddings, Chroma build/load
│   ├── rag.py            # prompt, retriever, ask(), history-aware query reformulation, @traceable
│   ├── machines.py       # machine registry + per-machine maintenance history lookup
│   ├── manuals.py        # per-page PDF extraction for the "download this page" feature
│   ├── agent.py          # Diagnostic Agent — tools, safety-first rule, tracing config, run_diagnostic_agent()
│   ├── services.py       # application service layer — a diagnostic turn as plain functions (no Streamlit)
│   ├── cost.py           # token metering callback, per-session usage, daily ledger + spend cap
│   ├── vision.py         # defect classification (trained + zero-shot), recommend_actions(), evaluate_classifier()
│   ├── defect_dataset.py # MVTec AD subset scan + stratified manifest
│   ├── tracing.py        # LangSmith project override + trace_config()/run_url() helpers
│   ├── safety.py         # Safety Validator — LLM-judge + keyword audit of safety-first ordering
│   └── evaluation.py     # eval_scenarios.csv loader + scoring + agent-vs-baseline pipeline runner
├── tests/                # pytest suite (unit + integration); see pyproject.toml, Makefile
├── data/
│   ├── manuals/
│   ├── defect_images/
│   └── vectorstore/
└── notebooks/
    ├── 01_data_ingestion.ipynb
    ├── 02_vector_database.ipynb
    ├── 03_rag_pipeline.ipynb
    ├── 04_conversation_history.ipynb
    ├── 05_maintenance_history.ipynb
    ├── 06_computer_vision.ipynb
    ├── 07_orchestrator_agent.ipynb
    ├── 08_tracing_observability.ipynb
    ├── 09_safety_validator.ipynb
    ├── 10_evaluation_baseline.ipynb
    ├── 11_retrieval_benchmark.ipynb
    └── 12_service_layer.ipynb
```

The notebooks and `app.py` both import their pipeline logic from `factory_floor/` — the notebooks stay for step-by-step exploration/demo, but no logic is duplicated between them and the Streamlit app. Since the professionalization work, `app.py` is a thin Streamlit view over `factory_floor/services.py`: it reads widgets and session state, calls `services.run_diagnostic(...)`, and renders the result.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your OpenAI API key:

```text
OPENAI_API_KEY=...
```

`.env` is listed in `.gitignore` and is never meant to be committed or shared.

## Step 1 — Download the manuals

```bash
python download_manuals.py
```

The source manifest uses official Siemens documentation links (mirrored via `cache.industry.siemens.com`, `docs.rs-online.com` or other official/verified sources where `support.industry.siemens.com/cs/attachments/` blocks automated and even manual browser downloads). If a link ever breaks, open the corresponding URL in `manual_sources.csv`, download it manually, and place the file in `data/manuals/` using the listed filename.

### Corpus (11 manuals, 3679 pages)

Balanced across both equipment families and multiple document types (operating instructions, list/fault manuals, function manuals, engineering manual, catalog). Each row in `manual_sources.csv` carries an explicit `equipment_type` column (`VFD` / `electric_motor`), which `factory_floor/ingestion.py` reads directly instead of guessing from the filename:

| Manual | Family | Document type | Equipment |
|---|---|---|---|
| Siemens_SINAMICS_G120C_Operating_Instructions.pdf | SINAMICS G120C | operating_manual | VFD |
| Siemens_SINAMICS_G120C_List_Manual.pdf | SINAMICS G120C | list_manual | VFD |
| Siemens_SINAMICS_G120_Function_Manual.pdf | SINAMICS G120 | function_manual | VFD |
| Siemens_CU240B2_CU240E2_Operating_Instructions.pdf | SINAMICS G120 CU240B-2/CU240E-2 | operating_manual | VFD |
| Siemens_G120_CU240BE2_List_Manual.pdf | SINAMICS G120 CU240B/E-2 | list_manual | VFD |
| Siemens_G120_Fieldbus_Function_Manual.pdf | SINAMICS G120 | function_manual | VFD |
| Siemens_SIMOTICS_SD_Operating_Instructions.pdf | SIMOTICS SD | operating_manual | electric motor |
| Siemens_SIMOTICS_GP_SD_DP_Engineering_Manual.pdf | SIMOTICS GP/SD/DP | engineering_manual | electric motor |
| Siemens_SIMOTICS_SD_1LE7_Operating_Instructions.pdf | SIMOTICS SD 1LE7 | operating_manual | electric motor |
| Siemens_SIMOTICS_GP_1LE1_Operating_Instructions.pdf | SIMOTICS GP 1LE1 | operating_manual | electric motor |
| Siemens_SIMOTICS_GP_SD_XP_DP_Catalog.pdf | SIMOTICS GP/SD/XP/DP | catalog | electric motor |

Page split: ~2766 pages VFD / ~913 pages electric motor (roughly 3:1 — down from an initial 18:1 imbalance in the earliest 4-manual draft).

## Machines & maintenance history (20 assets)

`machines.csv` is a simulated registry of 20 physical assets — 10 electric motors + 10 VFDs — each with a `machine_id`, `equipment_type` (same `VFD` / `electric_motor` vocabulary as `manual_sources.csv`), `family`, `model`, `location`, and `install_date`. `maintenance_history.csv` holds 2-6 simulated fault/repair/preventive-maintenance events per machine.

**VFD fault codes — 17 real, verified codes**, extracted directly from the raw text of the ingested List Manual PDFs (not guessed, not sampled from a semantic search — a full regex scan of every page, cross-checked against the vector store), each confirmed to retrieve a grounded RAG answer when paired with a short symptom keyword:

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

(An earlier draft used invented 4-digit codes like "F0051" that don't exist in the real manuals and returned no answer — fixed this session by scanning the actual PDF text for the real `F\d{5}`/`A\d{5}` pattern, 191 real F-codes found in total, then verifying a diverse 17-code subset against the vector store.)

**Electric motors have no numeric fault-code system at all** — confirmed by scanning all 5 motor manuals for the same code pattern: zero matches. Motor events use a free-text `description` grounded in real terminology from those manuals instead: abnormal/unusual noise, insulation resistance below the specified value, condensation in the terminal box, re-greasing per the lubricant plate interval, checking direction of rotation with the motor uncoupled, voltage/magnetic/mechanical imbalance.

Note: bare fault-code queries (just "F30021?" with no other words) retrieve poorly — the embedding-based retriever needs a short symptom description alongside the code to reliably find the right manual page, since these list-manual pages are dense with many similar-looking codes. `test_questions.txt` demonstrates the working phrasing pattern.

Both files were generated once with a fixed random seed and are committed as static data — same convention as `manual_sources.csv`, not regenerated at runtime. `factory_floor/machines.py` only reads them, via `load_machines()`, `load_maintenance_history()`, and `get_machine_history(machine_id)`.

In the Streamlit app, the operator picks a machine/VFD from a sidebar selector before asking a question. That selection (a) filters the RAG search to that machine's `equipment_type`, via the `equipment_type` parameter `factory_floor/rag.py`'s `build_retriever()` already supported, and (b) shows that machine's maintenance history in an expander. This is the "History Tool" from the target architecture — combining it with the RAG Tool inside a reasoning agent is a later roadmap step (Orchestrator Agent), not part of this milestone.

## Step 2 — Run the notebooks in order

Open Jupyter or VS Code and execute:

1. `notebooks/01_data_ingestion.ipynb`
2. `notebooks/02_vector_database.ipynb`
3. `notebooks/03_rag_pipeline.ipynb`
4. `notebooks/04_conversation_history.ipynb`
5. `notebooks/05_maintenance_history.ipynb`
6. `notebooks/06_computer_vision.ipynb`
7. `notebooks/07_orchestrator_agent.ipynb`
8. `notebooks/08_tracing_observability.ipynb`
9. `notebooks/09_safety_validator.ipynb`
10. `notebooks/10_evaluation_baseline.ipynb`

Notebook 02 creates the persistent Chroma database under `data/vectorstore/`. Notebook 04 demonstrates multi-turn follow-up questions on top of the same retriever, including how an elliptical follow-up gets rewritten into a standalone question before retrieval. Notebook 05 demonstrates the machine registry and per-machine maintenance history lookup (no LLM/vectorstore calls needed). Notebook 06 trains and benchmarks the defect classifier. Notebook 07 builds and verifies the Diagnostic Agent. Notebook 08 verifies real LangSmith traces (needs `LANGSMITH_API_KEY` in `.env` to actually see them in the UI, but runs and asserts fine without one). Notebook 09 builds and A/B-tests the Safety Validator. Notebook 10 runs the full 37-scenario evaluation, agent-vs-baseline benchmark, and safety audit (takes several minutes — dozens of real API calls).

## Step 3 — Launch Streamlit

```bash
streamlit run app.py
```

## What to show in the Saturday presentation

Demonstrate the pipeline in this order:

- Show the chosen domain: electric motors + VFDs.
- Show the official source PDFs / source manifest.
- Open notebook 01 and explain loading + metadata.
- Open notebook 02 and explain chunking, embeddings, Chroma and retrieval.
- Run one raw retrieval query and show the actual chunks/pages returned.
- Open notebook 03 and run an end-to-end RAG question.
- Show that the response contains source labels and that retrieved pages are traceable.
- Launch Streamlit and repeat the question through the UI, then ask a follow-up question to show the conversation continuing with context.
- Show the machine/VFD selector in the sidebar and the resulting maintenance history table, and how switching machines changes which manuals get searched.
- State explicitly that vision, agents, memory and safety validation are the next project phases.

## Important limitation

This milestone is **not yet the final diagnostic agent**. It is a grounded technical-document assistant. It should not be presented as an autonomous industrial safety or maintenance decision system.
