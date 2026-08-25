"""Builds fault_codes.csv — every fault/alarm code that actually appears in the ingested
manuals, with the page its definition is on.

Run once after the vector store is built (notebooks 01-02). The result is committed as
static reference data, the same convention as machines.csv and manual_sources.csv: small,
structured, generated once, human-inspectable, and read-only at runtime.

Why a committed file rather than scanning at startup: the app needs `known_codes()` to be
instant, and a reviewer being able to open the CSV and see all real codes is worth more
than saving one file.
"""

import re
from collections import defaultdict

from dotenv import load_dotenv

from factory_floor.config import COLLECTION_NAME, FAULT_CODES_CSV, VECTOR_DIR
from factory_floor.fault_codes import FAULT_CODE_PATTERN, save_code_index, score_occurrence
from factory_floor.vectorstore import get_embeddings, load_vectorstore

load_dotenv()

# A definition line runs to the end of its sentence/clause; keep it short and readable.
_DEFINITION_CUTOFF = re.compile(r"\s{2,}|\n|Message class:|Reaction:|Acknowledge:")


def clean_definition(context: str) -> str:
    text = _DEFINITION_CUTOFF.split(context.strip(), maxsplit=1)[0]
    return " ".join(text.split())[:120]


def build_index(vectorstore) -> list:
    """One pass over every chunk. For each code, keep the single best-scoring occurrence
    — the same best-match-per-code rule the 2026-08-18-d audit had to introduce, because
    a code's first appearance is usually an incidental cross-reference from another page,
    not its definition."""
    collection = vectorstore._collection
    total = collection.count()
    print(f"Scanning {total} chunks for {FAULT_CODE_PATTERN.pattern} ...")

    everything = collection.get(include=["documents", "metadatas"])
    best = {}
    occurrences = defaultdict(int)

    for text, metadata in zip(everything["documents"], everything["metadatas"]):
        for code in set(FAULT_CODE_PATTERN.findall(text)):
            occurrences[code] += 1
            score, context = score_occurrence(text, code)
            if code not in best or score > best[code][0]:
                best[code] = (score, metadata, context)

    rows = []
    for code in sorted(best):
        score, metadata, context = best[code]
        rows.append(
            {
                "code": code,
                "kind": "fault" if code.startswith("F") else "alarm",
                "source_file": metadata.get("source_file", "unknown"),
                "page": metadata.get("page", 0) + 1,
                "definition": clean_definition(context),
            }
        )

    faults = sum(1 for r in rows if r["kind"] == "fault")
    print(f"  {len(rows)} unique codes ({faults} F / {len(rows) - faults} A)")
    undefined = [r["code"] for r in rows if not r["definition"]]
    if undefined:
        print(f"  note: {len(undefined)} codes have no readable definition line "
              f"(cross-reference only), e.g. {undefined[:5]}")
    return rows


if __name__ == "__main__":
    vectorstore = load_vectorstore(VECTOR_DIR, COLLECTION_NAME, embeddings=get_embeddings())
    rows = build_index(vectorstore)
    save_code_index(rows)
    print(f"OK    written to {FAULT_CODES_CSV}")
