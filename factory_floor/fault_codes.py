"""Exact lookup for Siemens fault/alarm codes.

Embedding search cannot do this job, and no amount of tuning fixes it: an embedding
represents *meaning*, and "F30021" has none — it is a label. F30021 and F30022 land in
almost the same place in vector space despite being unrelated faults, so a bare-code
query retrieves whichever fault-list page is semantically "closest to a generic fault
question". Worse, the retriever always returns its k nearest neighbours with no
relevance floor, so an unknown code still yields real manual text about *a different
fault*, which the model then explains confidently, with citations. In maintenance that
is how someone applies the wrong procedure to real equipment.

So codes are looked up literally, not searched semantically. Chroma's own full-text
filter does the work (`where_document={"$contains": code}`, trigram-tokenised, so true
substring matching) — no extra dependency, no re-indexing.
"""

import csv
import difflib
import re
from pathlib import Path

from langchain_core.documents import Document

from factory_floor.config import FAULT_CODES_CSV

# Same pattern as the 2026-08-18-d corpus audit: Siemens SINAMICS uses 5-digit F (fault)
# and A (alarm) codes. The earlier 4-digit guesses (F0051 etc.) were invented and appear
# nowhere in the manuals — see CLAUDE.md 2026-08-18-c.
FAULT_CODE_PATTERN = re.compile(r"\b([FA]\d{5})\b")

# Operators read these codes off a converter display and re-type them, so the classic
# character confusions are the realistic failure mode, not random typos. difflib is the
# wrong tool for it — it scores "F3OO21" against "F30021" below any usable threshold
# because it sees O and 0 as unrelated characters (measured: no match even at cutoff
# 0.75). Normalising the confusions first turns the same case into an exact hit.
_DIGIT_CONFUSIONS = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})

# Loose pattern for a code typed with those confusions. Requiring at least 3 real digits
# stops it matching ordinary words — "FOSSIL" fits [FA][0-9OISBLQZ]{5} otherwise.
_MALFORMED_CODE_PATTERN = re.compile(r"\b([FA][0-9OQILSBZ]{5})\b", re.IGNORECASE)
_MIN_REAL_DIGITS = 3


def normalize_code(raw: str) -> str:
    """'F3OO21' -> 'F30021'. Letter-for-digit confusions only, and only in the numeric
    part; the leading F/A is left alone."""
    raw = raw.upper()
    return raw[:1] + raw[1:].translate(_DIGIT_CONFUSIONS)

# A code's real definition reads "F30021 Power unit: Ground fault ...". These are the
# subsystem prefixes that actually introduce one in this corpus.
_DEFINITION_PREFIX = re.compile(
    r"(Power unit|Drive|CU|PU|TM|SI|PROFIBUS|Infeed|Internal fan|Encoder|Motor)\b",
    re.IGNORECASE,
)

# ...whereas most occurrences of a code are incidental cross-references from other
# pages. Naively taking the first match is exactly the bug hit in 2026-08-18-e: F30021's
# first hit is "See also: F30021 Note: This parameter is only relevant for chassis power
# units", several hundred pages before its real definition.
_CROSS_REFERENCE_BEFORE = re.compile(
    r"(See also|Note|Notice|Danger|Warning|Description)\s*:?\s*$", re.IGNORECASE
)
_CODE_LIST_AFTER = re.compile(r"^[,\s]*[FA]\d{5}")

MANIFEST_FIELDS = ["code", "kind", "source_file", "page", "definition"]


def extract_codes(text: str) -> list:
    """Fault/alarm codes mentioned in a piece of text, in order, without duplicates."""
    if not text:
        return []
    seen = set()
    codes = []
    for code in FAULT_CODE_PATTERN.findall(text.upper()):
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def score_occurrence(text: str, code: str) -> tuple:
    """Best (score, following-text) for any occurrence of `code` in `text`.

    Verified against the real vectorstore on 8 known codes: picks the definition chunk
    8 times out of 8, including F07011 whose code appears in 53 different chunks."""
    best_score = -99
    best_context = ""
    for match in re.finditer(re.escape(code), text):
        before = text[max(0, match.start() - 60):match.start()].rstrip()
        after = text[match.end():match.end() + 200].lstrip()
        score = 0
        if _CROSS_REFERENCE_BEFORE.search(before):
            score -= 5
        if _CODE_LIST_AFTER.match(after):
            score -= 4
        if _DEFINITION_PREFIX.match(after):
            score += 6
        if re.match(r"^[A-Z][a-z]", after):
            score += 2
        if score > best_score:
            best_score = score
            best_context = after
    return best_score, best_context


def lookup_code(vectorstore, code: str, equipment_type: str = None, limit: int = 200) -> list:
    """Every chunk literally containing `code`, best-scoring (i.e. the definition) first.

    Goes straight to the Chroma collection rather than through the retriever: this is a
    lexical lookup, so embedding the query would be wasted work and would reintroduce
    the very fuzziness being avoided. Returns LangChain Documents so callers can treat
    the result exactly like retriever output — rag.format_context() and source_list()
    only ever read metadata['source_file'] and metadata['page']."""
    where = {"equipment_type": equipment_type} if equipment_type else None
    result = vectorstore._collection.get(
        where_document={"$contains": code},
        where=where,
        limit=limit,
        include=["documents", "metadatas"],
    )

    scored = []
    for text, metadata in zip(result["documents"], result["metadatas"]):
        score, _context = score_occurrence(text, code)
        scored.append((score, Document(page_content=text, metadata=dict(metadata))))

    scored.sort(key=lambda pair: -pair[0])
    return [doc for _score, doc in scored]


# --- Committed code index (same convention as machines.csv / manual_sources.csv) ----

def load_code_index(path: Path = FAULT_CODES_CSV) -> list:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_code_index(rows: list, path: Path = FAULT_CODES_CSV) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def known_codes(path: Path = FAULT_CODES_CSV) -> set:
    return {row["code"] for row in load_code_index(path)}


def get_code_entry(code: str, path: Path = FAULT_CODES_CSV) -> dict:
    return next((row for row in load_code_index(path) if row["code"] == code.upper()), None)


def suggest_similar(code: str, path: Path = FAULT_CODES_CSV, cutoff: float = 0.85) -> list:
    """Near-miss codes, for mistyping only — 'F3OO21' (letter O for zero) is a different
    situation from a code that genuinely does not exist, and only the first deserves a
    suggestion. Deliberately NOT used to soften the not-found guardrail: a suggestion is
    offered to the operator as a question to confirm, never fed to the model as if it
    were the code they asked about.

    Character-confusion normalisation runs first and, when it lands on a real code, wins
    outright — it is a far stronger signal than a fuzzy string score."""
    codes = known_codes(path)
    normalized = normalize_code(code)
    if normalized in codes:
        return [normalized]
    return difflib.get_close_matches(code.upper(), sorted(codes), n=3, cutoff=cutoff)


def extract_possible_codes(text: str) -> list:
    """Codes typed with character confusions that normalise onto a real code — the
    'F3OO21' case, which FAULT_CODE_PATTERN cannot match at all because the digit part
    contains letters. Returns (as_typed, normalised) pairs, and only when the normalised
    form is genuinely in the corpus, so this never invents a correction."""
    if not text:
        return []
    codes = known_codes()
    found = []
    seen = set()
    for raw in _MALFORMED_CODE_PATTERN.findall(text):
        raw = raw.upper()
        if FAULT_CODE_PATTERN.fullmatch(raw):
            continue  # already a well-formed code; not a typo
        if sum(ch.isdigit() for ch in raw[1:]) < _MIN_REAL_DIGITS:
            continue  # guards against ordinary words like "FOSSIL"
        normalized = normalize_code(raw)
        if normalized in codes and raw not in seen:
            seen.add(raw)
            found.append((raw, normalized))
    return found
