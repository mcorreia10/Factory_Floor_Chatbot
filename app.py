import itertools
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

from factory_floor import audit, identity, recurrence, services
from factory_floor.cache import SemanticCache
from factory_floor.config import MANUAL_DIR, VECTOR_DIR, get_settings
from factory_floor.cost import DailyLedger, UsageAccumulator
from factory_floor.machines import (
    OUTCOME_LABELS,
    append_resolution_event,
    get_machine_history,
    load_machines,
)
from factory_floor.manuals import extract_page_pdf
from factory_floor.rag import get_llm
from factory_floor.secrets import get_secret
from factory_floor.vision import CLASSIFIER_PATH, load_classifier

load_dotenv()
# factory_floor is imported above (which snapshots Settings) before load_dotenv() runs,
# so drop that pre-.env snapshot now — the first real get_settings() call below reads
# the loaded environment. See CLAUDE.md 2026-08-21 on why the import order is fixed.
get_settings.cache_clear()

st.set_page_config(page_title="The Factory Floor", layout="wide")

# The agent writes each section of its answer as a markdown level-4 heading (rule 7 of
# DIAGNOSTIC_SYSTEM_PROMPT) — "Safety precautions", "Remedies suggested by the manual",
# and so on. Style them here rather than asking the model to bold them by hand, so the
# look is consistent whether the text came from the agent, a safety rewrite, or the
# cache. h4 is not used anywhere else in this app (st.title/st.subheader emit h1/h3).
st.markdown(
    """
    <style>
      .stMarkdown h4 {
          font-size: 1.18rem;
          font-weight: 700;
          font-style: italic;
          margin-top: 1.1rem;
          margin-bottom: 0.35rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

LANGUAGES = {
    "🇬🇧 English": "English",
    "🇫🇷 Français": "French",
    "🇵🇹 Português": "Portuguese",
    "🇪🇸 Español": "Spanish",
    "🇩🇪 Deutsch": "German",
}

_, text_size_col, language_col = st.columns([2, 1, 1])
with text_size_col:
    text_size = st.selectbox(
        "Text size",
        ["Normal", "Large", "Extra large"],
        label_visibility="collapsed",
    )
with language_col:
    selected_language = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        label_visibility="collapsed",
    )
answer_language = LANGUAGES[selected_language]
text_scale = {"Normal": 100, "Large": 130, "Extra large": 160}[text_size]
st.markdown(f"<style>html {{ font-size: {text_scale}%; }}</style>", unsafe_allow_html=True)

api_key = get_secret("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to a .env file before launching the app.")
    st.stop()

# Optional operator sign-in (phase 5). Off by default — set FACTORY_FLOOR_REQUIRE_LOGIN=true
# to require it. On a real shop floor this is a badge scan / PIN pad / MES SSO, not a form.
if get_settings().require_login and "operator" not in st.session_state:
    st.title("The Factory Floor")
    st.subheader("Operator sign-in")
    with st.form("operator_login"):
        _op_id = st.text_input("Operator ID", placeholder="e.g. OP-1001")
        _pin = st.text_input("PIN", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            _operator = identity.authenticate(_op_id.strip(), _pin)
            if _operator:
                st.session_state["operator"] = _operator.as_dict()
                st.rerun()
            else:
                st.error("Unknown operator ID or wrong PIN.")
    st.stop()

operator = st.session_state.get("operator") or {}
_tenant_id = operator.get("tenant_id", get_settings().tenant_id)

if not VECTOR_DIR.exists() or not any(VECTOR_DIR.iterdir()):
    st.error(
        "Vector database not found. Run notebooks/01_data_ingestion.ipynb and "
        "notebooks/02_vector_database.ipynb first."
    )
    st.stop()

@st.cache_resource
def load_rag_components(tenant_id):
    # tenant_id is the cache key — a real multi-tenant deployment loads each tenant's
    # own collection (phase 7 seam); "default" resolves to the existing store.
    vectorstore = services.load_tenant_vectorstore(tenant_id)
    llm = get_llm()
    return vectorstore, llm

vectorstore, llm = load_rag_components(_tenant_id)

GENERAL_MACHINE = {
    "machine_id": "GENERAL",
    "equipment_type": "",
    "family": "All equipment",
    "model": "-",
    "location": "-",
    "install_date": "-",
}

machines = load_machines()
machine_labels = {"🌐 General question (search all manuals)": GENERAL_MACHINE}
machine_labels.update({f"{m['machine_id']} — {m['family']} ({m['location']})": m for m in machines})

with st.sidebar:
    if operator:
        st.caption(f"👤 {operator['name']} ({operator['operator_id']} · {operator['role']})")
        if st.button("Sign out", use_container_width=True):
            st.session_state.pop("operator", None)
            st.rerun()

    st.subheader("Machine / Asset")
    selected_label = st.selectbox("Select equipment", list(machine_labels.keys()))
    selected_machine = machine_labels[selected_label]
    st.session_state["selected_machine"] = selected_machine
    if selected_machine["machine_id"] == "GENERAL":
        st.caption("Searches across all manuals — no equipment_type filter, no per-machine history.")
    else:
        st.caption(f"{selected_machine['model']} · installed {selected_machine['install_date']}")

    _session_usage = UsageAccumulator.from_dict(st.session_state.get("usage"))
    if _session_usage.n_calls:
        st.caption(f"Session LLM cost: ${_session_usage.total_usd:.4f} · {_session_usage.n_calls} calls")
    _cap = get_settings().daily_spend_cap_usd
    if _cap:
        _spent_today = DailyLedger().today_total(_tenant_id)
        st.progress(
            min(_spent_today / _cap, 1.0) if _cap else 0.0,
            text=f"Today: ${_spent_today:.2f} / ${_cap:.2f} daily cap",
        )

    if get_settings().semantic_cache_enabled:
        try:
            _cache_n = SemanticCache().count()
        except Exception:
            _cache_n = 0
        st.caption(f"⚡ Answer cache: {_cache_n} entries")
        if st.button("Clear answer cache", use_container_width=True):
            SemanticCache().clear()
            st.rerun()


@st.cache_data(show_spinner=False)
def cached_page_pdf(source_file, page_number):
    return extract_page_pdf(MANUAL_DIR, source_file, page_number)


@st.cache_resource
def load_vision_components():
    if not CLASSIFIER_PATH.exists():
        return None
    return load_classifier()


def source_rows(docs):
    rows = []
    # The code-not-found notice travels with the documents so the model sees it, but it
    # is an instruction, not a manual page — listing it as a source would put a fake row
    # under an answer that correctly says nothing was found.
    for i, doc in enumerate([d for d in docs if not d.metadata.get("not_found")], 1):
        rows.append(
            {
                "source": f"SOURCE {i}",
                "file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", 0) + 1,
                "equipment": doc.metadata.get("equipment_type", "unknown"),
            }
        )
    return rows


def render_prior_occurrences(report):
    """The zero-cost panel: what this machine's own records already say about this fault.

    Everything here is read from maintenance_history.csv and resolution_events — no model
    was called, so nothing on screen can be invented. It deliberately stops at reporting:
    ranking the past actions would recommend whatever was done most often, which on real
    data is the wrong advice (a fault answered twice with a module replacement and three
    times with a power cycle is a recurring hardware fault, not a power-cycle problem)."""
    code = report["fault_code"]
    count = report["count"]
    st.info(
        f"**{code} has happened on {report['machine_id']} before — "
        f"{count} previous occurrence{'s' if count > 1 else ''}.** "
        "Taken straight from this machine's records. No model was called, so this cost nothing."
    )

    rows = [
        {
            "Date": a["date"],
            "What was done": a["action"] or "—",
            "By": a["who"] or "—",
            "Outcome": OUTCOME_LABELS.get(a["outcome"], a["outcome"] or "not recorded"),
            "Source": "operator note" if a["source"] == "operator" else "logbook",
            "Downtime (h)": a["downtime_hours"] or "—",
        }
        for a in report["actions"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if report["recurring"]:
        gap = report["shortest_gap_days"]
        if report["returned_quickly"]:
            st.warning(
                f"⚠ **Recurring, and it came back fast** — only {gap} days between two "
                "occurrences. Whatever was done last time did not hold."
            )
        elif gap is not None:
            st.warning(
                f"⚠ **This fault has returned before** — shortest gap between occurrences "
                f"was {gap} days. A repair that was done already is not proof it is fixed."
            )
    if not report["outcomes"]:
        st.caption(
            "None of these records say whether the fix actually worked — the outcome field "
            "was added recently. Filling it in when you record a resolution is what will "
            "eventually make it possible to tell an effective action from a frequent one."
        )


MAILTO_MAX_CHARS = 1800


def _resolution_mailto(machine_id, operator, turn, steps_text):
    """A mailto: URL with the resolution report pre-filled, for the operator to send to
    their supervisor from their own mail client.

    Capped at MAILTO_MAX_CHARS: mail clients and browsers silently truncate or refuse
    very long mailto URLs, and a report that arrives cut in half is worse than one that
    says plainly it was shortened."""
    body_parts = [
        f"Machine: {machine_id}",
        f"Operator: {operator.get('name', 'unknown')} ({operator.get('operator_id', '-')})",
        f"Reported (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        "",
        "Question asked:",
        (turn.get("question") or "-").strip(),
        "",
        "Resolution steps taken:",
        (steps_text or "").strip() or "-",
        "",
        "-- Sent from The Factory Floor maintenance copilot.",
    ]
    body = "\n".join(body_parts)
    if len(body) > MAILTO_MAX_CHARS:
        body = body[:MAILTO_MAX_CHARS] + "\n\n[... shortened — see the machine history for the full record]"
    subject = f"Maintenance report — {machine_id}"
    return f"mailto:?subject={quote(subject)}&body={quote(body)}"


def describe_tool_call(entry):
    if entry["tool"] == "search_manuals":
        return f'🔍 Searched manuals for: "{entry["input"].get("query", "")}"'
    if entry["tool"] == "get_maintenance_history":
        return "📋 Checked this machine's maintenance history"
    return f"🔧 Called {entry['tool']}({entry['input']})"


st.session_state.setdefault("turns", [])


def render_turn(turn, turn_index):
    st.markdown(f"**Q{turn_index + 1}.** {turn['question']}")

    if turn.get("image_bytes"):
        img_col, result_col = st.columns([1, 2])
        with img_col:
            st.image(turn["image_bytes"], caption="Uploaded photo", use_container_width=True)
        with result_col:
            if turn.get("is_defective"):
                st.warning(f"Photo condition: **not good** — {turn['predicted_label'].replace('_', ' ')} detected")
            else:
                st.success("Photo condition: **good** — no defect detected")

    st.subheader("Diagnostic reasoning")
    if turn.get("cache_hit"):
        st.caption("⚡ Answered from the cache — an equivalent question was asked before.")
    st.markdown(turn["answer"])

    safety = turn.get("safety") or {}
    if safety.get("action") == "held":
        st.error("🛑 This answer was withheld by the safety gate — it recommended physical work "
                 "without stating precautions first. The safe fallback is shown above.")
    elif safety.get("action") == "rewritten":
        st.caption("🛡️ Safety precautions were added to this answer before it was shown.")

    tool_trace = turn.get("tool_trace") or []
    if tool_trace:
        st.caption("Tools used:")
        for entry in tool_trace:
            st.caption(describe_tool_call(entry))
    else:
        st.caption("Tools used: none — the agent answered directly, without consulting manuals or history.")

    used_manuals = any(entry["tool"] == "search_manuals" for entry in tool_trace)
    if not used_manuals:
        st.caption(
            "⚠️ This answer did not consult the equipment manuals — general guidance only. "
            "Cross-check against the manuals or escalate to a qualified technician for anything "
            "safety-critical."
        )

    if any(not d.metadata.get("not_found") for d in (turn.get("documents") or [])):
        st.subheader("Sources retrieved")
        st.dataframe(source_rows(turn["documents"]), use_container_width=True, hide_index=True)

        with st.expander("Show retrieved evidence"):
            for i, doc in enumerate([d for d in turn["documents"] if not d.metadata.get("not_found")], 1):
                source = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page", 0) + 1
                st.markdown(f"### {source}, page {page}")
                st.write(doc.page_content)
                st.download_button(
                    "Download this page (PDF)",
                    data=cached_page_pdf(source, page),
                    file_name=f"{Path(source).stem}_p{page}.pdf",
                    mime="application/pdf",
                    key=f"dl_turn{turn_index}_source_{i}",
                )

    st.divider()


def submit_turn(question_text, uploaded_photo):
    question_text = (question_text or "").strip()
    if not question_text and uploaded_photo is None:
        st.warning("Enter a maintenance question or upload a photo first.")
        return

    # A mistyped code is a different situation from an unknown one, and only the operator
    # can tell them apart — so ask, rather than silently correcting and answering about a
    # code they never asked about. Runs before any API call is made.
    if not st.session_state.pop("code_typo_confirmed", False):
        typos = services.check_typo(question_text)
        if typos:
            as_typed, suggestion = typos[0]
            st.session_state["pending_typo"] = {
                "question": question_text,
                "as_typed": as_typed,
                "suggestion": suggestion,
            }
            return

    # Has this exact fault already happened on this machine? Pure lookup over records we
    # already hold — no model, no cost — so the operator sees it before deciding whether
    # a diagnosis is worth running at all. Only on the first question of a conversation:
    # interrupting a follow-up would break the thread the operator is already in.
    if (
        not st.session_state["turns"]
        and uploaded_photo is None
        and not st.session_state.pop("prior_occurrence_ack", False)
    ):
        report = recurrence.prior_occurrence_report(
            st.session_state["selected_machine"]["machine_id"], question_text
        )
        if report:
            st.session_state["pending_prior"] = {"question": question_text, "report": report}
            return

    classification = None
    vision_context = None
    image_bytes = None
    if uploaded_photo is not None:
        vision_components = load_vision_components()
        if vision_components is None:
            st.warning("Defect classifier not trained yet — run notebooks/06_computer_vision.ipynb first.")
            return
        clf, _label_list = vision_components
        image_bytes = uploaded_photo.getvalue()
        with st.spinner("Looking at the photo..."):
            photo = services.classify_photo(image_bytes, clf)
        classification = photo["classification"]
        vision_context = photo["vision_context"]

    selected_machine = st.session_state["selected_machine"]
    request = services.DiagnosticRequest(
        question_text=question_text,
        machine_id=selected_machine["machine_id"],
        equipment_type=selected_machine["equipment_type"],
        chat_history=services.build_chat_history(st.session_state["turns"]),
        vision_context=vision_context,
        language=answer_language,
        operator_id=operator.get("operator_id"),
        tenant_id=_tenant_id,
    )

    st.markdown(
        f"**Q{len(st.session_state['turns']) + 1}.** "
        f"{question_text or '[Uploaded a photo of a component]'}"
    )

    generator, result = services.run_diagnostic(request, vectorstore=vectorstore, llm=llm, stream=True)

    if result.blocked:
        st.error(result.message)
        return

    with st.spinner("Reasoning about the evidence..."):
        first_chunk = next(generator, None)

    st.subheader("Diagnostic reasoning")
    if first_chunk is not None:
        st.write_stream(itertools.chain([first_chunk], generator))

    safety = result.safety or {}
    if safety.get("action") == "held":
        st.error("🛑 This answer was withheld by the safety gate — it recommended physical work "
                 "without stating precautions first. A safe fallback is shown above.")
    elif safety.get("action") == "rewritten":
        st.info("Safety precautions were added to this answer before it was shown.")

    turn = services.assemble_turn(
        result,
        image_bytes=image_bytes,
        classification=classification,
        vision_context=vision_context,
        language=answer_language,
    )
    st.session_state["turns"].append(turn)

    if result.cost:
        session_usage = UsageAccumulator.from_dict(st.session_state.get("usage"))
        session_usage.merge(result.cost)
        st.session_state["usage"] = session_usage.as_dict()

    st.session_state["followup_key"] = st.session_state.get("followup_key", 0) + 1
    st.rerun()


form_col, _spacer_col = st.columns([2, 1])

with form_col:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #2c6187 100%);
            border-radius: 16px;
            padding: 2.4rem 2rem;
            margin-bottom: 1.8rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.28);
        ">
            <svg viewBox="0 0 220 220" width="220" height="220"
                 style="position:absolute; top:-45px; right:-45px; opacity:0.10;">
                <g transform="translate(110,110)">
                    <circle r="70" fill="none" stroke="white" stroke-width="10"/>
                    <circle r="30" fill="none" stroke="white" stroke-width="8"/>
                    <g fill="white">
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(0)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(30)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(60)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(90)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(120)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(150)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(180)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(210)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(240)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(270)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(300)"/>
                        <rect x="-8" y="-100" width="16" height="24" transform="rotate(330)"/>
                    </g>
                </g>
            </svg>
            <svg viewBox="0 0 160 160" width="160" height="160"
                 style="position:absolute; bottom:-55px; left:-35px; opacity:0.08;">
                <g transform="translate(80,80)">
                    <circle r="50" fill="none" stroke="white" stroke-width="8"/>
                    <g fill="white">
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(0)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(45)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(90)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(135)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(180)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(225)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(270)"/>
                        <rect x="-6" y="-72" width="12" height="18" transform="rotate(315)"/>
                    </g>
                </g>
            </svg>
            <div style="position:relative; z-index:1;">
                <div style="font-size:2rem; font-weight:800; color:#ffffff; letter-spacing:-0.01em; line-height:1.25;">
                    Industrial Maintenance Copilot
                </div>
                <div style="font-size:1.1rem; font-weight:500; color:#c7d5e3; margin-top:0.35rem;">
                    Electric Motors + Variable-Frequency Drives
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state["turns"]:
        question = st.text_area(
            "Describe the maintenance problem",
            height=120,
            placeholder="Example: The motor is overheating and vibrating. What checks are supported by the manuals?",
        )
        st.caption("You can also attach a photo of the affected component (optional):")
        uploaded_photo = st.file_uploader(
            "Upload a photo of a cable, metal nut, screw, or transistor",
            type=["png", "jpg", "jpeg"],
            key="defect_photo_uploader",
        )
        if st.button("Search manuals and answer", type="primary", use_container_width=True):
            submit_turn(question, uploaded_photo)

    pending = st.session_state.get("pending_typo")
    if pending:
        st.warning(
            f"**{pending['as_typed']}** is not a code in these manuals, but "
            f"**{pending['suggestion']}** is — the two are easy to confuse when reading a "
            f"converter display. Which did you mean?"
        )
        confirm_col, keep_col = st.columns(2)
        with confirm_col:
            if st.button(f"Yes, I meant {pending['suggestion']}", type="primary", use_container_width=True):
                corrected = pending["question"].replace(pending["as_typed"], pending["suggestion"])
                st.session_state["code_typo_confirmed"] = True
                st.session_state.pop("pending_typo")
                submit_turn(corrected, None)
        with keep_col:
            if st.button(f"No, it really is {pending['as_typed']}", use_container_width=True):
                st.session_state["code_typo_confirmed"] = True
                st.session_state.pop("pending_typo")
                submit_turn(pending["question"], None)

    pending_prior = st.session_state.get("pending_prior")
    if pending_prior:
        render_prior_occurrences(pending_prior["report"])
        run_col, skip_col = st.columns(2)
        with run_col:
            if st.button("Run the full diagnosis anyway", type="primary", use_container_width=True):
                st.session_state["prior_occurrence_ack"] = True
                st.session_state.pop("pending_prior")
                submit_turn(pending_prior["question"], None)
        with skip_col:
            if st.button("That's enough — don't run it", use_container_width=True):
                st.session_state.pop("pending_prior")
                st.rerun()

for idx, turn in enumerate(st.session_state["turns"]):
    render_turn(turn, idx)

if st.session_state["turns"]:
    machine_id = st.session_state["selected_machine"]["machine_id"]
    if machine_id != "GENERAL":
        with st.expander(f"Maintenance history — {machine_id}"):
            history = get_machine_history(machine_id, include_resolutions=True)
            if history:
                st.dataframe(history, use_container_width=True, hide_index=True)
            else:
                st.caption("No recorded history for this machine.")

        _last_turn = st.session_state["turns"][-1]
        with st.expander("📝 Record what you actually did (adds to this machine's history)"):
            _steps = st.text_area(
                "Resolution steps taken",
                key=f"resolution_{st.session_state.get('followup_key', 0)}",
                placeholder="e.g. Isolated the drive, measured insulation resistance motor-to-earth (0.2 MΩ), "
                "replaced the motor cable, retested. Fault cleared.",
            )
            # The fault code is what makes this record findable the next time the same
            # fault appears; the outcome is what stops "done most often" being mistaken
            # for "actually worked". Both were missing until now, which is exactly why
            # earlier recorded resolutions cannot be matched to a fault at all.
            _code_col, _outcome_col = st.columns(2)
            with _code_col:
                _fault_code = st.text_input(
                    "Fault code this resolves (optional)",
                    value=recurrence.code_in_question(_last_turn.get("question", "")) or "",
                    key=f"resolution_code_{st.session_state.get('followup_key', 0)}",
                    placeholder="e.g. F30805",
                )
            with _outcome_col:
                _outcome_label = st.selectbox(
                    "Did it work?",
                    ["— not recorded —"] + list(OUTCOME_LABELS.values()),
                    key=f"resolution_outcome_{st.session_state.get('followup_key', 0)}",
                )
            _outcome = next(
                (k for k, v in OUTCOME_LABELS.items() if v == _outcome_label), ""
            )
            _save_col, _cmms_col, _mail_col = st.columns(3)
            with _save_col:
                if st.button("Save to machine history", use_container_width=True):
                    if not _steps.strip():
                        st.warning("Type what you did first.")
                    else:
                        _ev_id = append_resolution_event(
                            machine_id,
                            operator_id=operator.get("operator_id"),
                            steps_text=_steps.strip(),
                            recommendation_id=_last_turn.get("audit_id"),
                            fault_code=_fault_code,
                            outcome=_outcome,
                        )
                        st.session_state["last_resolution_id"] = _ev_id
                        st.success("Saved to this machine's history.")
                        st.rerun()
            with _cmms_col:
                if st.button(
                    "Send to CMMS/ERP (demo)",
                    use_container_width=True,
                    disabled="last_resolution_id" not in st.session_state,
                ):
                    _ack = audit.export_to_cmms(st.session_state["last_resolution_id"])
                    st.toast(f"CMMS accepted — ref {_ack['cmms_ref']}")
            with _mail_col:
                # A mailto: link, deliberately: it opens the operator's own mail client
                # with the report pre-filled, so there is no SMTP server to configure and
                # no mail credentials for this app to hold. The operator picks the
                # recipient and presses send, which is also the honest trust boundary —
                # the app never sends mail on someone's behalf.
                st.link_button(
                    "Email to supervisor",
                    _resolution_mailto(machine_id, operator, _last_turn, _steps),
                    use_container_width=True,
                    disabled=not _steps.strip(),
                )
            st.caption(
                "“Email to supervisor” opens your mail app with the report filled in — "
                "choose the recipient and send it yourself."
            )

    followup_col, _followup_spacer_col = st.columns([2, 1])
    with followup_col:
        followup_question = st.text_area(
            "Ask a follow-up question",
            height=100,
            placeholder="Example: What if that parameter is already set correctly?",
            key=f"followup_question_{st.session_state.get('followup_key', 0)}",
        )
        if st.button("Send follow-up", type="primary", use_container_width=True):
            submit_turn(followup_question, None)

    if st.button("← Back to start", use_container_width=True):
        st.session_state["turns"] = []
        st.rerun()
