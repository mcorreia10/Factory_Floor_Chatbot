"""Retrieval-only ablation: which chunking / search strategy / reranker setting actually
retrieves best?

Different question from notebooks/10_evaluation_baseline.ipynb. That one asks "is my
system better than the obvious alternative?" (agent vs. baseline). This one asks "why
this configuration and not another?" — the project has always used chunk_size=800,
search_type="similarity" and a reranker without any of those three choices ever being
measured.

Scored without a single LLM call. Every row of eval_scenarios.csv carries an
`expected_evidence_keywords` phrase that was individually verified to exist in the target
chunk's Cause/Remedy text and to be absent from the question itself, so checking whether a
retrieved chunk contains it is a real chunk-level hit test — no generation, no judging, no
cost, and no LLM noise masking the differences between configurations.
"""

import shutil
import time
from collections import defaultdict
from pathlib import Path

from factory_floor.config import COLLECTION_NAME, MANUAL_DIR, PROJECT_ROOT
from factory_floor.evaluation import load_eval_scenarios  # noqa: F401  (re-exported for notebooks)
from factory_floor.ingestion import load_manuals
from factory_floor.vectorstore import build_vectorstore, chunk_documents, get_embeddings, load_vectorstore

VARIANT_DIR_TEMPLATE = str(PROJECT_ROOT / "data" / "vectorstore_cs{chunk_size}")


def first_hit_rank(documents: list, keywords: list):
    """1-based position of the first retrieved chunk containing any expected evidence
    phrase, or None. Rank matters as much as presence: the model weights what comes first,
    so a configuration that finds the right chunk but buries it fifth is worse than one
    that leads with it — which is exactly the difference hit-rate alone cannot see."""
    if not keywords:
        return None
    for position, doc in enumerate(documents, 1):
        content = (doc.page_content or "").lower()
        if any(kw.lower() in content for kw in keywords):
            return position
    return None


def evaluate_retriever(retriever_factory, scenarios: list, verbose: bool = False) -> dict:
    """`retriever_factory(equipment_type) -> retriever` is a callable rather than a single
    retriever because each scenario carries its own equipment_type filter, exactly as
    evaluation.run_baseline() already does — comparing configurations without honouring
    that filter would measure something the app never actually runs.

    Reciprocal rank is 1/position (1st -> 1.00, 2nd -> 0.50, absent -> 0), so MRR rewards
    ranking the right chunk highly, while hit_rate only asks whether it was there at all."""
    per_category = defaultdict(lambda: {"n": 0, "hits": 0, "rr": 0.0})
    results = []
    hits = 0
    reciprocal_rank_total = 0.0
    started = time.time()

    for scenario in scenarios:
        retriever = retriever_factory(scenario["equipment_type"] or None)
        documents = retriever.invoke(scenario["question"])
        rank = first_hit_rank(documents, scenario["expected_evidence_keywords"])
        reciprocal = (1.0 / rank) if rank else 0.0

        hits += 1 if rank else 0
        reciprocal_rank_total += reciprocal

        bucket = per_category[scenario["category"]]
        bucket["n"] += 1
        bucket["hits"] += 1 if rank else 0
        bucket["rr"] += reciprocal

        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "category": scenario["category"],
                "rank": rank,
                "reciprocal_rank": reciprocal,
                "n_documents": len(documents),
            }
        )
        if verbose:
            print(f"  [{scenario['scenario_id']}] rank={rank if rank else '-':>3}  "
                  f"({len(documents)} docs)")

    n = len(scenarios) or 1
    return {
        "n_scenarios": len(scenarios),
        "hit_rate": hits / n,
        "mrr": reciprocal_rank_total / n,
        "elapsed_s": time.time() - started,
        "per_category": {
            name: {
                "n": b["n"],
                "hit_rate": b["hits"] / b["n"] if b["n"] else 0.0,
                "mrr": b["rr"] / b["n"] if b["n"] else 0.0,
            }
            for name, b in sorted(per_category.items())
        },
        "results": results,
    }


def variant_dir(chunk_size: int) -> Path:
    return Path(VARIANT_DIR_TEMPLATE.format(chunk_size=chunk_size))


def build_variant_store(chunk_size: int, chunk_overlap: int = None, manual_dir=None,
                        embeddings=None, force: bool = False):
    """Builds (or loads, if already present) a Chroma store for one chunking variant.

    Each variant gets its OWN directory. This is not tidiness: build_vectorstore's
    rebuild=True does shutil.rmtree() on the directory it is given, so pointing a variant
    at the production VECTOR_DIR — even with a different collection_name — would delete
    the real store. Idempotent by design so a notebook re-run costs nothing.
    """
    chunk_overlap = chunk_overlap if chunk_overlap is not None else int(chunk_size * 0.15)
    target = variant_dir(chunk_size)
    embeddings = embeddings or get_embeddings()

    if target.exists() and not force:
        return load_vectorstore(target, COLLECTION_NAME, embeddings=embeddings)

    if force and target.exists():
        shutil.rmtree(target)

    documents = load_manuals(manual_dir or MANUAL_DIR)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  chunk_size={chunk_size} overlap={chunk_overlap} -> {len(chunks)} chunks "
          f"(from {len(documents)} pages)")
    return build_vectorstore(chunks, target, COLLECTION_NAME, embeddings=embeddings, rebuild=True)


def variant_chunk_count(store) -> int:
    return store._collection.count()


def format_table(title: str, rows: list, label_width: int = 34) -> str:
    """Fixed-width printed table, matching notebook 10's presentation style (no pandas,
    no plots) so the two evaluation notebooks read as one body of work.
    `rows` is a list of (label, hit_rate, mrr, extra)."""
    lines = [title, "-" * (label_width + 34)]
    lines.append(f"{'Configuration':<{label_width}}{'hit-rate@k':>12}{'MRR':>10}{'':>12}")
    for label, hit_rate, mrr, extra in rows:
        lines.append(f"{label:<{label_width}}{hit_rate:>11.1%}{mrr:>10.3f}{extra:>12}")
    return "\n".join(lines)
