"""Streamlit chat UI for the Enterprise Support Agent, with document upload and sample data."""

import logging

import streamlit as st

from src.agent import run_agent
from src.ingestion import (
    SUPPORTED,
    build_index,
    delete_document,
    ensure_index,
    list_documents,
    save_upload,
)

st.set_page_config(page_title="Enterprise Support Agent", page_icon="🤖")

# Logger that prints to the terminal; the handlers check stops duplicate lines on Streamlit reruns
log = logging.getLogger("support_agent")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

NEW_DEPT = "➕ New department..."
EXAMPLES = [  # clickable demo questions (they match the sample documents)
    "How long do refunds take?",
    "Where is my order ORD-1001?",
    "How many paid leaves do I get?",
    "Do you offer a student discount?",
    "I am furious, let me talk to a manager",
    "What is the capital of France?",
]


# Index sample and uploaded documents once per app start if the vector DB is empty
@st.cache_resource(show_spinner="Preparing knowledge base (first run only)...")
def prepare_knowledge_base() -> int:
    return ensure_index()


# Write the route, agent path and any ticket to the terminal log (not shown to the user)
def log_result(result: dict) -> None:
    log.info(
        "question=%r | route=%s | path=%s",
        result["question"],
        result.get("route"),
        " -> ".join(result["path"]),
    )
    if result.get("ticket_id"):
        log.warning(
            "ticket created: %s | question=%r", result["ticket_id"], result["question"]
        )


# Show sources to everyone; show ticket, agent path and retrieved chunks only in developer mode
def render_details(result: dict, show_details: bool = False) -> None:
    if result.get("sources"):
        st.caption("📄 Sources: " + ", ".join(result["sources"]))
    if not show_details:
        return  # normal users stop here
    if result.get("ticket_id"):
        st.caption(f"🎫 Ticket: {result['ticket_id']}")
    with st.expander("Agent path"):
        st.write(" → ".join(result["path"]))
    if result.get("contexts"):
        with st.expander(f"Retrieved chunks ({len(result['contexts'])})"):
            for chunk in result["contexts"]:
                st.text(chunk[:300])


# Sidebar panel: choose files and a department, then index them
def render_upload_panel(departments: list[str]) -> None:
    st.subheader("📚 Add your documents")
    key = st.session_state.setdefault(
        "uploader_key", 0
    )  # changing the key resets the uploader
    files = st.file_uploader(
        "PDF, Markdown or text files",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED)],
        accept_multiple_files=True,
        key=f"uploader_{key}",
    )
    choice = st.selectbox("Department", departments + [NEW_DEPT])
    department = st.text_input("New department name") if choice == NEW_DEPT else choice

    if st.button("Add to knowledge base", disabled=not files or not department):
        added = 0
        with st.spinner("Reading and indexing..."):
            for f in files:
                chunks = save_upload(f.name, f.getvalue(), department)
                if chunks:
                    added += chunks
                else:
                    st.warning(f"No readable text found in {f.name}")
        st.session_state.uploader_key += 1
        st.session_state.flash = f"Added {len(files)} file(s), {added} chunks indexed"
        st.rerun()


# Sidebar list of indexed documents; uploads get a delete button, samples are protected
def render_document_list(docs: list[dict]) -> None:
    with st.expander(f"Indexed documents ({len(docs)})"):
        if not docs:
            st.caption("Nothing indexed yet.")
        for d in docs:
            col_name, col_btn = st.columns([4, 1])
            tag = " (sample)" if d["origin"] == "sample" else ""
            col_name.caption(
                f"{d['department']} / {d['source']}{tag} · {d['chunks']} chunks"
            )
            if d["origin"] == "upload" and col_btn.button(
                "🗑", key=f"del_{d['department']}_{d['source']}"
            ):
                delete_document(d["source"], d["department"])
                st.rerun()


# Draw the sidebar, replay the chat history and handle a new question
def main() -> None:
    st.title("🤖 Enterprise Support Agent")
    st.caption("Ask about your documents or an order status.")

    prepare_knowledge_base()
    docs = list_documents()
    has_uploads = any(d["origin"] == "upload" for d in docs)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # chat history survives Streamlit reruns here

    with st.sidebar:
        if "flash" in st.session_state:
            st.success(st.session_state.pop("flash"))

        use_samples = st.checkbox(
            "Try with sample data (NovaCart documents)",
            value=not has_uploads,  # default on until the user uploads something
            key="use_samples",
        )
        show_details = st.checkbox("Show agent details (developer mode)", value=False)

        # Only documents that are currently searchable are shown and filterable
        visible = [d for d in docs if use_samples or d["origin"] == "upload"]
        visible_departments = sorted({d["department"] for d in visible})
        all_departments = sorted({d["department"] for d in docs})

        st.divider()
        render_upload_panel(all_departments)
        render_document_list(visible)
        st.divider()

        department = st.selectbox(
            "Search in", ["All departments"] + visible_departments
        )
        if use_samples:
            st.subheader("Try these")
            for example in EXAMPLES:
                if st.button(example, use_container_width=True):
                    st.session_state.pending = example
        if st.button("Rebuild index"):
            with st.spinner("Re-indexing all documents..."):
                count = build_index(reset=True)
            st.session_state.flash = f"Re-indexed {count} chunks"
            st.rerun()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if not visible:
        st.info(
            "Nothing to search yet. Upload a document or tick 'Try with sample data'."
        )

    for msg in st.session_state.messages:  # replay earlier messages
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("result"):
                render_details(msg["result"], show_details)

    question = st.chat_input("Ask a question...") or st.session_state.pop(
        "pending", None
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                dept = None if department == "All departments" else department
                result = run_agent(question, dept, include_samples=use_samples)
            log_result(result)
            st.write(result["answer"])
            render_details(result, show_details)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "result": result}
        )


main()
