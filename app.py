import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from factory_floor.agent import run_diagnostic_agent
from factory_floor.config import COLLECTION_NAME, MANUAL_DIR, VECTOR_DIR
from factory_floor.machines import get_machine_history, load_machines
from factory_floor.manuals import extract_page_pdf
from factory_floor.rag import build_chat_history, build_retriever, get_llm
from factory_floor.vectorstore import get_embeddings, load_vectorstore
from factory_floor.vision import CLASSIFIER_PATH, classify_defect_trained, load_classifier

load_dotenv()

st.set_page_config(page_title="The Factory Floor", page_icon="🏭", layout="wide")

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

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to a .env file before launching the app.")
    st.stop()

if not VECTOR_DIR.exists() or not any(VECTOR_DIR.iterdir()):
    st.error(
        "Vector database not found. Run notebooks/01_data_ingestion.ipynb and "
        "notebooks/02_vector_database.ipynb first."
    )
    st.stop()

@st.cache_resource
def load_rag_components():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(VECTOR_DIR, COLLECTION_NAME, embeddings=embeddings)
    llm = get_llm()
    return vectorstore, llm

vectorstore, llm = load_rag_components()

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
    st.subheader("Machine / Asset")
    selected_label = st.selectbox("Select equipment", list(machine_labels.keys()))
    selected_machine = machine_labels[selected_label]
    st.session_state["selected_machine"] = selected_machine
    if selected_machine["machine_id"] == "GENERAL":
        st.caption("Searches across all manuals — no equipment_type filter, no per-machine history.")
    else:
        st.caption(f"{selected_machine['model']} · installed {selected_machine['install_date']}")


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
    for i, doc in enumerate(docs, 1):
        rows.append(
            {
                "source": f"SOURCE {i}",
                "file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", 0) + 1,
                "equipment": doc.metadata.get("equipment_type", "unknown"),
            }
        )
    return rows


def describe_tool_call(entry):
    if entry["tool"] == "search_manuals":
        return f'🔍 Searched manuals for: "{entry["input"].get("query", "")}"'
    if entry["tool"] == "get_maintenance_history":
        return "📋 Checked this machine's maintenance history"
    return f"🔧 Called {entry['tool']}({entry['input']})"


TOP_K = 5

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
    st.markdown(turn["answer"])

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

    if turn.get("documents"):
        st.subheader("Sources retrieved")
        st.dataframe(source_rows(turn["documents"]), use_container_width=True, hide_index=True)

        with st.expander("Show retrieved evidence"):
            for i, doc in enumerate(turn["documents"], 1):
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
            classification = classify_defect_trained(io.BytesIO(image_bytes), clf)
        vision_context = (
            f"Vision analysis of the uploaded photo: predicted condition = "
            f"{classification['predicted_label']} (confidence {classification['confidence']:.0%}, "
            f"{'defective' if classification['is_defective'] else 'no defect detected'})."
        )

    retriever = build_retriever(
        vectorstore, k=TOP_K, equipment_type=st.session_state["selected_machine"]["equipment_type"]
    )
    chat_history = build_chat_history(st.session_state["turns"])
    machine_id = st.session_state["selected_machine"]["machine_id"]

    with st.spinner("Reasoning about the evidence..."):
        result = run_diagnostic_agent(
            question_text,
            retriever,
            machine_id,
            chat_history=chat_history,
            vision_context=vision_context,
            llm=llm,
            language=answer_language,
        )

    turn = {
        "type": "agent",
        "question": question_text or "[Uploaded a photo of a component]",
        "answer": result["answer"],
        "documents": result["documents"],
        "sources": result["sources"],
        "tool_trace": result["tool_trace"],
        "image_bytes": image_bytes,
        "predicted_label": classification["predicted_label"] if classification else None,
        "is_defective": classification["is_defective"] if classification else None,
        "language": answer_language,
    }
    st.session_state["turns"].append(turn)
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
                    <span style="margin-right:0.5rem;">🏭</span>Industrial Maintenance Copilot
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

for idx, turn in enumerate(st.session_state["turns"]):
    render_turn(turn, idx)

if st.session_state["turns"]:
    machine_id = st.session_state["selected_machine"]["machine_id"]
    if machine_id != "GENERAL":
        with st.expander(f"Maintenance history — {machine_id}"):
            history = get_machine_history(machine_id)
            if history:
                st.dataframe(history, use_container_width=True, hide_index=True)
            else:
                st.caption("No recorded history for this machine.")

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
