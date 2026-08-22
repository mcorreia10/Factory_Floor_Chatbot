import time

from factory_floor.agent import run_diagnostic_agent
from factory_floor.config import EVAL_SCENARIOS_CSV
from factory_floor.machines import get_machine_history
from factory_floor.rag import ask, build_retriever

KEYWORD_PASS_RATE = 0.5  # arbitrary threshold, disclosed rather than hidden -- see notebook 10


def load_eval_scenarios(path=EVAL_SCENARIOS_CSV) -> list:
    """Reads eval_scenarios.csv, splitting '|'-separated multi-value columns into
    lists. `expected_source_pattern` is normalized to None when blank (the `general`
    category rows leave it empty on purpose -- a GENERAL question must not be scored
    against any single manual family)."""
    import csv

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scenarios = []
    for row in rows:
        scenarios.append(
            {
                "scenario_id": row["scenario_id"],
                "category": row["category"],
                "machine_id": row["machine_id"] or "GENERAL",
                "equipment_type": row["equipment_type"],
                "question": row["question"],
                "expected_root_cause_keywords": [
                    k.strip() for k in row["expected_root_cause_keywords"].split("|") if k.strip()
                ],
                "expected_evidence_keywords": [
                    k.strip() for k in row["expected_evidence_keywords"].split("|") if k.strip()
                ],
                "expected_source_pattern": row["expected_source_pattern"].strip() or None,
                "notes": row["notes"],
            }
        )
    return scenarios


def keyword_score(text: str, keywords: list) -> dict:
    """Case-insensitive substring match -- a proxy for correctness, not human grading.
    An answer phrased differently from these keywords scores as a miss even if it is
    right; an answer that echoes the keywords without being right scores as a hit.
    Disclosed as a limitation in notebook 10, not hidden."""
    text_lower = (text or "").lower()
    hits = [kw for kw in keywords if kw.lower() in text_lower]
    missing = [kw for kw in keywords if kw not in hits]
    rate = (len(hits) / len(keywords)) if keywords else 1.0
    return {"hits": hits, "missing": missing, "n": len(keywords), "rate": rate, "passed": rate >= KEYWORD_PASS_RATE}


def source_score(documents: list, pattern) -> bool:
    """Checks the *manual family* (a substring of source_file), not the exact page --
    no per-page human ground truth exists for this evaluation set. Returns None when
    `pattern` is falsy (the `general`-category rows), meaning this check is skipped
    for that scenario rather than scored as a failure."""
    if not pattern:
        return None
    return any(pattern in (doc.metadata.get("source_file") or "") for doc in documents)


def history_reference_score(answer: str, machine_id: str):
    """Checks whether the answer mentions one of this machine's own recorded history
    *dates* (e.g. "2022-06-25") from maintenance_history.csv. Returns None for a
    GENERAL question (no machine to have history for) rather than False -- there is
    nothing to reference, so it isn't a failure to not reference it.

    Deliberately checks only the event date, not the fault_code: an earlier version of
    this function also matched on fault_code, but every eval scenario's question
    already names its own fault code, so a plain RAG answer discussing that code (with
    no history awareness at all) matched trivially -- caught via a smoke test on V01
    (a fault code that also happens to appear in VFD-01's simulated history) before
    this notebook was finalized. The date is a much stronger signal: nothing about the
    question or the manual content would cause a date like this to appear by chance.

    rag.ask() has no access to per-machine history at all, so its score here is
    structurally 0 for every non-GENERAL scenario -- notebook 10 asserts this, since a
    nonzero baseline score on this metric would mean the metric itself is leaking."""
    if not machine_id or machine_id == "GENERAL":
        return None
    rows = get_machine_history(machine_id)
    if not rows:
        return False
    answer_text = answer or ""
    return any(row.get("event_date") and row["event_date"] in answer_text for row in rows)


def run_baseline(scenario: dict, vectorstore, llm) -> dict:
    """The plain single-call pipeline (rag.ask()) -- the "simpler alternative"
    benchmarked against the agent. Builds the retriever identically to run_agent() so
    the only variable between the two arms is the pipeline architecture, not the
    retrieval configuration."""
    retriever = build_retriever(vectorstore, k=5, equipment_type=scenario["equipment_type"])
    result = ask(scenario["question"], retriever, llm=llm)
    return {"answer": result["answer"], "documents": result["documents"], "run_id": result["run_id"]}


def run_agent(scenario: dict, vectorstore, llm) -> dict:
    """The Diagnostic Agent (run_diagnostic_agent()) -- the main design being
    evaluated."""
    retriever = build_retriever(vectorstore, k=5, equipment_type=scenario["equipment_type"])
    result = run_diagnostic_agent(scenario["question"], retriever, machine_id=scenario["machine_id"], llm=llm)
    return {
        "answer": result["answer"],
        "documents": result["documents"],
        "run_id": result["run_id"],
        "tool_trace": result["tool_trace"],
    }


def evaluate_pipeline(scenarios: list, runner, vectorstore, llm, limit: int = None, verbose: bool = True) -> dict:
    """Runs `runner` (run_baseline or run_agent) over every scenario and scores each
    on 4 axes, mirrors vision.py's evaluate_classifier() shape. Accuracy fields are
    None when no scenario in the set has a non-None value for that axis (e.g.
    source_accuracy when every scenario in a filtered subset is `general`)."""
    scenarios = scenarios[:limit] if limit else scenarios
    results = []
    for scenario in scenarios:
        start = time.monotonic()
        run_result = runner(scenario, vectorstore, llm)
        latency = time.monotonic() - start

        root_cause = keyword_score(run_result["answer"], scenario["expected_root_cause_keywords"])
        evidence = keyword_score(run_result["answer"], scenario["expected_evidence_keywords"])
        source = source_score(run_result["documents"], scenario["expected_source_pattern"])
        history = history_reference_score(run_result["answer"], scenario["machine_id"])
        n_tool_calls = len(run_result.get("tool_trace") or [])

        row = {
            "scenario_id": scenario["scenario_id"],
            "category": scenario["category"],
            "question": scenario["question"],
            "answer": run_result["answer"],
            "run_id": run_result.get("run_id"),
            "root_cause_passed": root_cause["passed"],
            "evidence_passed": evidence["passed"],
            "source_match": source,
            "history_match": history,
            "latency_s": latency,
            "n_tool_calls": n_tool_calls,
        }
        results.append(row)
        if verbose:
            print(
                f"[{scenario['scenario_id']}] root_cause={root_cause['passed']} evidence={evidence['passed']} "
                f"source={source} history={history} ({latency:.1f}s, {n_tool_calls} tool calls)"
            )

    n = len(results)

    def _rate(key):
        vals = [r[key] for r in results if r[key] is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    return {
        "n_scenarios": n,
        "root_cause_accuracy": _rate("root_cause_passed"),
        "evidence_accuracy": _rate("evidence_passed"),
        "source_accuracy": _rate("source_match"),
        "history_accuracy": _rate("history_match"),
        "mean_latency_s": (sum(r["latency_s"] for r in results) / n) if n else 0.0,
        "mean_tool_calls": (sum(r["n_tool_calls"] for r in results) / n) if n else 0.0,
        "results": results,
    }
