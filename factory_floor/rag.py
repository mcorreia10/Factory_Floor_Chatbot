from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langsmith import traceable

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
    messages = []
    for turn in turns[-max_turns:]:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    return messages


def build_retriever(vectorstore, k=5, equipment_type=None):
    kwargs = {"k": k}
    if equipment_type:
        kwargs["filter"] = {"equipment_type": equipment_type}
    return vectorstore.as_retriever(search_kwargs=kwargs)


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
