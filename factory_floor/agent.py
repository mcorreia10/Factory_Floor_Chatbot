from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from factory_floor.machines import get_machine, get_machine_history
from factory_floor.rag import LLM_MODEL, format_context, get_llm
from factory_floor.tracing import trace_config

DIAGNOSTIC_SYSTEM_PROMPT = """You are the diagnostic reasoning component of an industrial
maintenance copilot for electric motors and variable-frequency drives (VFDs).

You have tools available:
- search_manuals(query): searches the equipment manuals and returns relevant excerpts,
  each labelled with its source file and page number.
- get_maintenance_history(): returns this specific machine's past fault/repair/
  preventive-maintenance history. Only offered when a specific machine is selected, not
  for general questions.

Alongside the operator's message, you may also be given the result of a vision analysis
of an uploaded photo (the predicted condition of a component), and the registry details
of the specific machine the operator selected. Combine everything available — the
described symptom, the vision analysis if present, the machine details, and the
machine's history if you consult it — into a single, coherent root-cause hypothesis and
concrete next steps, safety precautions first.

When the machine's registry details are given, never ask the operator what kind of
equipment it is, which model it is, or where it is installed — that is already
established fact for this request, and re-asking it wastes the operator's time on
information the system already has.

Rules:
1. Never invent a fault code, parameter, procedure, or manual reference. Only cite what
   a tool actually returned.
1b. FAULT CODES. If the operator's message contains a fault or alarm code such as
   F30021 or A30015, you MUST call search_manuals with that exact code, spelled exactly
   as the operator wrote it, before answering anything about it. Never paraphrase the
   code away or search only for a described symptom instead — the exact string is what
   makes the lookup reliable. If a tool result says a code was NOT FOUND in the manuals,
   that is final: say plainly that this code is not documented in the available manuals
   and ask the operator to re-check it on the equipment display. In that case you must
   not describe the code, must not guess its meaning, and must not substitute a
   similar-looking code — answering about a different fault than the one the operator is
   facing is worse than admitting the code is unknown.
2. When you cite manual content, cite it by file name and page number exactly as given
   in the tool result (e.g. "Siemens SINAMICS G120C List Manual, page 214"), not by a
   transient source number — you may call search_manuals more than once, and source
   numbers are not stable across calls.
3. If you have not called search_manuals, say explicitly that your answer is general
   guidance, not sourced from this equipment's documentation.
4. If the operator's described symptom and the vision analysis of the photo point to
   different problems, or if you do not have enough information for a confident
   hypothesis, ask ONE concise clarifying question instead of guessing. This applies any
   time you have genuine doubt, not only when signals visibly conflict.
5. Respond in {language}. Never translate fault codes, parameter numbers, or equipment
   model names — operators must recognize them on the physical equipment display.
6. SAFETY-FIRST OUTPUT CONTRACT — this rule outranks every rule above it. If your
   answer contains ANY step a person would physically perform on the equipment
   (inspect, measure, open, disconnect, reset, re-torque, clean, replace, restart),
   the answer MUST begin with a short "Safety precautions" section placed BEFORE the
   first such step — never after it, never only at the end. Ground those precautions
   in the safety instructions the manuals themselves contain: isolate and de-energize
   the drive or motor, apply lockout/tagout, wait for the DC link capacitors to
   discharge before touching any terminal, verify absence of voltage, and note that
   the work is for qualified personnel only. If you do not already have the
   equipment's safety instructions, call search_manuals for them. If the tool returns
   nothing usable, still state the precautions and say explicitly that they were not
   retrieved from this equipment's documentation. If your answer contains no physical
   action at all — you are asking a clarifying question, or only explaining what a
   fault code means — no safety section is required and you must not pad the answer
   with one.
7. This is an educational milestone, not yet a certified safety system."""


def format_machine_context(machine_id: str) -> str:
    """The selected machine's registry details, as a line the agent sees up front.

    Without this the agent knows the machine only indirectly — machine_id is used to
    filter retrieval and to close over the history tool, but is never stated in the
    conversation — so it would ask the operator what equipment type/model this is,
    information the sidebar selection already established. Returns "" for a general
    question (no machine selected) or an unknown id, in which case asking IS the
    correct behavior."""
    if not machine_id or machine_id == "GENERAL":
        return ""
    machine = get_machine(machine_id)
    if not machine:
        return ""
    equipment = "a variable-frequency drive (VFD)" if machine.get("equipment_type") == "VFD" else "an electric motor"
    return (
        f"Machine selected by the operator: {machine['machine_id']} — {equipment}, "
        f"{machine.get('family', '?')} (model {machine.get('model', '?')}), "
        f"installed {machine.get('install_date', '?')} at {machine.get('location', '?')}."
    )


def format_history(rows: list) -> str:
    """Formats raw CSV rows from machines.get_machine_history() into bullets readable
    by the LLM — analogous to rag.format_context(), but for maintenance history instead
    of manual chunks."""
    if not rows:
        return "No recorded maintenance history for this machine."
    lines = []
    for row in rows:
        label = row.get("fault_code") or row.get("description", "")
        lines.append(
            f"- {row.get('event_date', '?')} [{row.get('event_type', '?')}] {label} "
            f"— action taken: {row.get('action_taken', '?')}"
        )
    return "\n".join(lines)


def source_list_from_docs(docs: list) -> str:
    """Same shape/spirit as rag.source_list(), but built from documents accumulated
    across possibly multiple search_manuals tool calls within one agent run, not a
    single retriever.invoke()."""
    seen = set()
    lines = []
    for doc in docs:
        if doc.metadata.get("not_found"):
            continue  # the code-not-found notice is an instruction, not a citable source
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", 0) + 1
        key = (source, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"- {source} — page {page}")
    return "\n".join(lines)


def build_rag_tool(retriever):
    """Returns (search_manuals tool, retrieved_docs). retrieved_docs is a list that
    fills up via closure every time the tool is called during one agent invocation —
    the caller reads it back afterward to rebuild the 'Sources retrieved' UI and PDF
    downloads, exactly as before. Deliberately thin (retrieval + formatting only, no
    LLM synthesis) so the agent — not a second sub-LLM call — does the final citing and
    combining with vision/history evidence."""
    retrieved_docs = []

    @tool
    def search_manuals(query: str) -> str:
        """Search the equipment manuals for troubleshooting information. Use a specific
        query combining the fault code or symptom keywords, e.g. 'F30021 ground fault'
        rather than just a bare code — bare codes retrieve poorly."""
        docs = retriever.invoke(query)
        retrieved_docs.extend(docs)
        return format_context(docs) if docs else "No relevant manual content found for this query."

    return search_manuals, retrieved_docs


def build_history_tool(machine_id: str):
    """machine_id is closed over, never an LLM-controlled argument — it comes from the
    sidebar machine selector, a fact of the request context, not something the model
    should choose or risk inventing."""

    @tool
    def get_maintenance_history() -> str:
        """Get this machine's past fault/repair/preventive-maintenance history."""
        rows = get_machine_history(machine_id)
        return format_history(rows)

    return get_maintenance_history


def build_diagnostic_agent(retriever, machine_id: str, llm=None):
    """Assembles the available tools (search_manuals always; get_maintenance_history
    only when a specific machine is selected) and returns (compiled_agent,
    retrieved_docs, tools) — retrieved_docs is read by the caller after invoke(), tools
    lets the caller report which tools were actually available for this run (e.g. for
    trace metadata) without reconstructing the same machine_id logic a second time."""
    llm = llm or get_llm()
    search_manuals, retrieved_docs = build_rag_tool(retriever)
    tools = [search_manuals]
    if machine_id and machine_id != "GENERAL":
        tools.append(build_history_tool(machine_id))
    agent = create_agent(model=llm, tools=tools)
    return agent, retrieved_docs, tools


def _extract_tool_trace(messages: list) -> list:
    """Walks the message list a compiled agent returns and pairs each tool call with
    its result, for display in the UI ('Tools used') without needing to open LangSmith."""
    trace = []
    pending = {}
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                pending[call["id"]] = {"tool": call["name"], "input": call["args"]}
        elif isinstance(message, ToolMessage):
            entry = pending.get(message.tool_call_id, {"tool": message.name, "input": {}})
            entry["output_preview"] = str(message.content)[:300]
            trace.append(entry)
    return trace


def _build_agent_run(
    question_text: str,
    retriever,
    machine_id: str,
    chat_history: list,
    vision_context: str,
    llm,
    language: str,
    config: dict,
):
    """Shared setup for both the blocking (run_diagnostic_agent) and streaming
    (stream_diagnostic_agent) entry points, so a blocking call and a streamed call
    for the 'same' request build an identical agent, message list, and run config.

    `config` is an optional RunnableConfig-shaped dict forwarded to the agent; a
    caller can pass e.g. {"tags": [...]} to add to the defaults below, or nothing at
    all (the default trace_config already tags/names every run so it shows up
    meaningfully in LangSmith without any caller effort)."""
    llm = llm or get_llm()
    chat_history = chat_history or []
    agent, retrieved_docs, tools = build_diagnostic_agent(retriever, machine_id, llm=llm)

    human_parts = []
    machine_context = format_machine_context(machine_id)
    if machine_context:
        human_parts.append(machine_context)
    if question_text and question_text.strip():
        human_parts.append(f"Operator's described symptom: {question_text.strip()}")
    if vision_context:
        human_parts.append(vision_context)
    human_content = "\n\n".join(human_parts) if human_parts else "(no additional information provided)"

    system_message = SystemMessage(DIAGNOSTIC_SYSTEM_PROMPT.format(language=language))
    messages = [system_message] + chat_history + [HumanMessage(content=human_content)]

    default_config = trace_config(
        "diagnostic_agent",
        tags=["diagnostic_agent", f"machine:{machine_id or 'GENERAL'}"],
        metadata={
            "machine_id": machine_id or "GENERAL",
            "language": language,
            "llm_model": LLM_MODEL,
            "has_vision_context": bool(vision_context),
            "n_history_messages": len(chat_history),
            "tools_available": [t.name for t in tools],
        },
    )
    run_config = {**default_config, **(config or {})}

    return agent, retrieved_docs, messages, run_config


def run_diagnostic_agent(
    question_text: str,
    retriever,
    machine_id: str,
    chat_history: list = None,
    vision_context: str = None,
    llm=None,
    language: str = "English",
    config: dict = None,
) -> dict:
    """Single entry point, same spirit as rag.ask(). question_text and/or
    vision_context may be provided (an operator can submit a symptom, a photo, or
    both) — at least one is expected by the caller."""
    agent, retrieved_docs, messages, run_config = _build_agent_run(
        question_text, retriever, machine_id, chat_history, vision_context, llm, language, config,
    )

    result = agent.invoke({"messages": messages}, config=run_config)
    answer = result["messages"][-1].content
    tool_trace = _extract_tool_trace(result["messages"])

    return {
        "question": question_text,
        "answer": answer,
        "documents": retrieved_docs,
        "sources": source_list_from_docs(retrieved_docs),
        "tool_trace": tool_trace,
        "language": language,
        "run_id": str(run_config["run_id"]),
    }


class StreamedAgentRun:
    """Populated once the generator returned by stream_diagnostic_agent() is fully
    consumed (by st.write_stream() in app.py, or "".join(gen)/list(gen) in a
    notebook/test — this class has no Streamlit dependency). Same information as
    run_diagnostic_agent()'s returned dict, just delivered as attributes after the
    fact instead of as a dict up front, since 'answer' isn't known until generation
    finishes."""

    def __init__(self, question_text, language, run_id):
        self.question = question_text
        self.language = language
        self.run_id = run_id
        self.answer = None
        self.documents = []
        self.sources = None
        self.tool_trace = None


def stream_diagnostic_agent(
    question_text: str,
    retriever,
    machine_id: str,
    chat_history: list = None,
    vision_context: str = None,
    llm=None,
    language: str = "English",
    config: dict = None,
):
    """Streaming counterpart to run_diagnostic_agent() — identical inputs/setup (via
    _build_agent_run), but returns (generator, result) instead of a finished dict.
    generator yields str chunks of ONLY real, user-visible answer text; result is a
    StreamedAgentRun left empty until the generator is exhausted by the caller.

    agent.stream({"messages": messages}, config=run_config, stream_mode=["messages",
    "values"]) yields (mode, payload) pairs: "messages" pairs with (message_chunk,
    metadata) token deltas, "values" pairs with the graph's cumulative state (the
    last one is shape-identical to what agent.invoke() returns). During tool-call
    decisions, AIMessageChunk.content is always '' (the JSON args live only in
    .tool_call_chunks) and the tool's own result flows through as a full ToolMessage,
    not a chunk. The `langgraph_node == "model"` check is what makes that filter safe
    once a tool itself calls an LLM: rag.py's reranker runs inside search_manuals, so
    its ranking output ("0,14,13,...") arrives as content-bearing AIMessageChunks too —
    but under node "tools", not "model" (verified empirically against this graph, not
    assumed). Without the node check those digits stream straight to the operator."""
    agent, retrieved_docs, messages, run_config = _build_agent_run(
        question_text, retriever, machine_id, chat_history, vision_context, llm, language, config,
    )
    result = StreamedAgentRun(question_text, language, str(run_config["run_id"]))

    def _generate():
        final_values = None
        for mode, payload in agent.stream(
            {"messages": messages}, config=run_config, stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                message_chunk, metadata = payload
                if (
                    isinstance(message_chunk, AIMessageChunk)
                    and message_chunk.content
                    and metadata.get("langgraph_node") == "model"
                ):
                    yield message_chunk.content
            elif mode == "values":
                final_values = payload
        final_messages = final_values["messages"]
        result.answer = final_messages[-1].content
        result.documents = retrieved_docs
        result.sources = source_list_from_docs(retrieved_docs)
        result.tool_trace = _extract_tool_trace(final_messages)

    return _generate(), result
