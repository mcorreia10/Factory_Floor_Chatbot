import re
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langsmith import traceable

from factory_floor.fault_codes import extract_codes, lookup_code

LLM_MODEL = "gpt-4.1-mini"
MAX_HISTORY_TURNS = 4

SYSTEM_PROMPT = """You are the retrieval component of an industrial maintenance copilot for electric motors and variable-frequency drives.

Rules:
1. Answer only from the retrieved technical context.
2. Do not invent fault codes, limits, procedures, causes, or safety instructions.
3. If the context is insufficient, say that the documentation retrieved is insufficient.
4. Give a concise troubleshooting explanation with relevant checks, evidence and what should be verified next.
5. Cite evidence inline using the exact [SOURCE n] labels present in the context.
6. Respond in {language}. Never translate fault codes, parameter numbers, or equipment/model names —
   keep them exactly as they appear in the source documentation, since operators must recognize them
   on the physical equipment display.
7. This is an educational milestone and not yet the final diagnostic agent."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}"),
])

CONTEXTUALIZE_SYSTEM_PROMPT = """Given the conversation so far and the operator's latest follow-up
question, rewrite the follow-up into a standalone question that contains all context needed to
retrieve the right documentation. Preserve technical terminology, fault codes, part/parameter
numbers, and equipment names exactly as used. Do not answer the question — only rewrite it. If the
follow-up is already standalone, return it unchanged."""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def contextualize_question(question, chat_history, llm=None):
    if not chat_history:
        return question
    llm = llm or get_llm()
    messages = CONTEXTUALIZE_PROMPT.invoke({"question": question, "chat_history": chat_history})
    return llm.invoke(messages).content.strip()


def build_chat_history(turns, max_turns=MAX_HISTORY_TURNS):
    """Replays each turn's question and answer as messages the LLM can see on the next
    call. A photo turn's vision_context (e.g. 'predicted condition = deformation...')
    is folded back in here too — without this, a follow-up question loses all memory
    of a photo classified earlier in the conversation, since the turn's stored
    'question' is only the operator's typed text, never the vision result."""
    messages = []
    for turn in turns[-max_turns:]:
        human_parts = []
        if turn.get("question"):
            human_parts.append(turn["question"])
        if turn.get("vision_context"):
            human_parts.append(turn["vision_context"])
        human_content = "\n\n".join(human_parts) if human_parts else "(no additional information provided)"
        messages.append(HumanMessage(content=human_content))
        messages.append(AIMessage(content=turn["answer"]))
    return messages


RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a relevance-ranking component for a technical manual search system. Given a "
               "query and a numbered list of candidate excerpts, return ONLY a comma-separated list of "
               "the excerpt numbers, ordered from most to least relevant to the query. Include every "
               "number exactly once. No explanation, no other text."),
    ("human", "QUERY: {query}\n\nCANDIDATES:\n{candidates}"),
])


def rerank_documents(query, docs, llm=None, top_n=5):
    """Re-scores a wider candidate pool by true relevance to the query via one LLM call —
    unlike Chroma's raw vector distance, this can recover a chunk that mentions the exact
    fault code but wasn't the semantically closest neighbor (dificuldades_e_oportunidades.md,
    difficulty #1). Fails open: any parsing problem falls back to the original vector-search
    order, never raises."""
    if len(docs) <= top_n:
        return docs
    llm = llm or get_llm()
    candidates = "\n\n".join(f"[{i}] {doc.page_content[:600]}" for i, doc in enumerate(docs))
    messages = RERANK_PROMPT.format_messages(query=query, candidates=candidates)
    response = llm.invoke(messages)
    try:
        order = [int(tok.strip()) for tok in response.content.split(",") if tok.strip().lstrip("-").isdigit()]
        seen = set()
        ranked = []
        for i in order:
            if 0 <= i < len(docs) and i not in seen:
                ranked.append(docs[i])
                seen.add(i)
        ranked += [doc for i, doc in enumerate(docs) if i not in seen]
        return ranked[:top_n]
    except Exception:
        return docs[:top_n]


class RerankingRetriever:
    """Duck-types the .invoke(query) interface every caller in this project already uses
    (agent.py's search_manuals tool, rag.py's ask()) — fetches a wider candidate pool from
    the base retriever, then narrows it back to top_n via rerank_documents(). Opt-in via
    build_retriever(..., rerank=True); every existing caller that doesn't pass it keeps the
    exact previous behavior."""

    def __init__(self, base_retriever, llm, top_n):
        self.base_retriever = base_retriever
        self.llm = llm
        self.top_n = top_n

    def invoke(self, query):
        candidates = self.base_retriever.invoke(query)
        return rerank_documents(query, candidates, llm=self.llm, top_n=self.top_n)


CODE_NOT_FOUND_NOTICE = (
    "NOT FOUND: fault code {code} does not appear anywhere in the ingested manuals. "
    "Do not describe this code, and do not infer what it might mean from similar-looking "
    "codes. State plainly that it is not documented in the available manuals and ask the "
    "operator to re-check the code on the equipment display."
)


class CodeAwareRetriever:
    """Routes queries containing a fault code (F30021, A30015, ...) through an exact
    lexical lookup, and everything else through the normal semantic path untouched.

    Duck-types .invoke(query) like RerankingRetriever, which is all agent.py's
    search_manuals tool consumes.

    Two properties matter more than the retrieval itself:
      * exact hits are PINNED at the top and never handed to the reranker — an LLM
        reordering them could bury the one chunk that actually defines the code;
      * a code that yields zero literal hits produces an explicit not-found Document
        rather than silently falling back to semantic neighbours. That fallback is the
        dangerous case: it returns real manual text about a *different* fault, which
        reads as a well-cited answer to the wrong question."""

    def __init__(self, vectorstore, semantic_retriever, k, equipment_type=None):
        self.vectorstore = vectorstore
        self.semantic_retriever = semantic_retriever
        self.k = k
        self.equipment_type = equipment_type

    @staticmethod
    def _has_searchable_text(query, codes, min_words=3):
        """Is there anything worth a semantic search once the codes are removed? 'F99999'
        alone is not; 'F99999, motor overheating and tripping' is."""
        stripped = query
        for code in codes:
            stripped = re.sub(re.escape(code), " ", stripped, flags=re.IGNORECASE)
        return len([w for w in re.findall(r"[A-Za-z]{3,}", stripped)]) >= min_words

    def invoke(self, query):
        codes = extract_codes(query)
        if not codes:
            return self.semantic_retriever.invoke(query)

        # A bare code is answered entirely by the literal hits; there is nothing else in
        # the question for a semantic search to be about. Padding the result out to k
        # actively hurts: a bare code's nearest semantic neighbours are pages dense with
        # unrelated alphanumerics — motor catalogue nameplate rows like
        # "1LE15433AB434AA4-Z ... 6209-2ZC3" score as "similar" to "F30021" precisely
        # because of the weakness this class exists to work around.
        has_text = self._has_searchable_text(query, codes)
        per_code = 2 if has_text else max(1, self.k // len(codes))

        pinned = []
        seen = set()
        for code in codes:
            hits = lookup_code(self.vectorstore, code, equipment_type=self.equipment_type)
            if not hits:
                pinned.append(
                    Document(
                        page_content=CODE_NOT_FOUND_NOTICE.format(code=code),
                        metadata={
                            "source_file": "(not in corpus)",
                            "page": -1,
                            "fault_code": code,
                            "not_found": True,
                        },
                    )
                )
                continue
            for doc in hits[:per_code]:  # definition first, then its cause/remedy runner-up
                key = (doc.metadata.get("source_file"), doc.metadata.get("page"))
                if key not in seen:
                    seen.add(key)
                    pinned.append(doc)

        # Covers both "bare code, found it" (the literal hits are the whole answer) and
        # "bare code, not in the corpus" (padding would put real manual pages about
        # unrelated faults under an answer that correctly says nothing was found).
        if not has_text:
            return pinned[: self.k]

        remaining = self.k - len(pinned)
        if remaining <= 0:
            return pinned[: self.k]

        # The operator described something as well, so semantic results genuinely add
        # value (plain-language versions in the Operating Instructions, related
        # symptoms) — they just must not outrank the definition. Dedup against each
        # other too, not only against the pinned hits: two chunks of the same page share
        # a (file, page) key and would otherwise both be listed as separate sources.
        semantic = []
        for doc in self.semantic_retriever.invoke(query):
            key = (doc.metadata.get("source_file"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            semantic.append(doc)
            if len(semantic) >= remaining:
                break
        return pinned + semantic


def build_retriever(vectorstore, k=5, equipment_type=None, rerank=False, rerank_llm=None,
                    fetch_k=None, code_aware=False, search_type="similarity",
                    search_kwargs_extra=None):
    """`search_type` and `search_kwargs_extra` exist for the ablation study in
    notebooks/11_retrieval_benchmark.ipynb — "similarity" plus no extras reproduces the
    exact previous behaviour, so no existing caller changes. "mmr" takes `lambda_mult`
    and "similarity_score_threshold" takes `score_threshold` via search_kwargs_extra."""
    fetch_k = fetch_k or max(k * 3, 12)
    kwargs = {"k": fetch_k if rerank else k}
    if equipment_type:
        kwargs["filter"] = {"equipment_type": equipment_type}
    if search_type == "mmr":
        # MMR needs its own candidate pool to diversify from; without this it has
        # nothing to trade relevance against.
        kwargs.setdefault("fetch_k", max(kwargs["k"] * 4, 20))
    if search_kwargs_extra:
        kwargs.update(search_kwargs_extra)
    base = vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)
    if rerank:
        base = RerankingRetriever(base, rerank_llm or get_llm(), top_n=k)
    if code_aware:
        return CodeAwareRetriever(vectorstore, base, k=k, equipment_type=equipment_type)
    return base


def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", 0) + 1
        blocks.append(f"[SOURCE {i}] {source}, page {page}\n{doc.page_content}")
    return "\n\n".join(blocks)


def source_list(docs):
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", 0) + 1
        lines.append(f"- [SOURCE {i}] {source} — page {page}")
    return "\n".join(lines)


@traceable(name="rag_baseline", run_type="chain")
def _ask_traced(question, retriever, llm, chat_history, language):
    standalone_question = contextualize_question(question, chat_history, llm)
    docs = retriever.invoke(standalone_question)
    context = format_context(docs)
    messages = PROMPT.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history,
            "language": language,
        }
    )
    answer = llm.invoke(messages).content
    return {
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": source_list(docs),
        "documents": docs,
        "language": language,
    }


def ask(question, retriever, llm=None, chat_history=None, language="English", config: dict = None):
    """The plain single-call RAG pipeline — this is the baseline the evaluation
    notebook benchmarks the agent against, and deliberately carries no safety-first
    prompt rule (see factory_floor/agent.py), so the safety audit's comparison between
    the two is a real measurement, not a tautology.

    ask() is plain Python (contextualize -> retrieve -> prompt -> llm.invoke), not a
    single Runnable, so it uses @traceable instead of passing config= to each
    sub-call — that keeps one root run per ask() call instead of producing several
    unrelated sibling root traces, which would make it an unfair comparison against
    the agent's single grouped run."""
    llm = llm or get_llm()
    chat_history = chat_history or []
    run_id = (config or {}).get("run_id") or uuid4()
    extra = {
        "run_id": run_id,
        "tags": ["factory_floor", "rag_baseline"],
        "metadata": {
            "component": "rag_baseline",
            "language": language,
            "llm_model": LLM_MODEL,
            "pipeline": "plain_single_call",
        },
    }
    extra.update({k: v for k, v in (config or {}).items() if k != "run_id"})
    result = _ask_traced(question, retriever, llm, chat_history, language, langsmith_extra=extra)
    result["run_id"] = str(run_id)
    return result
