"""Streamlit interface for the local AI College Study Assistant."""

import streamlit as st

from backend.rag import AssistantError, LocalStudyAssistant
from config.settings import get_settings
from database.repository import StudyRepository


@st.cache_resource
def get_services() -> tuple[StudyRepository, LocalStudyAssistant]:
    settings = get_settings()
    repository = StudyRepository(settings.data_dir / "study_assistant.sqlite3")
    assistant = LocalStudyAssistant(settings, repository)
    return repository, assistant


def render_sidebar(repository: StudyRepository, assistant: LocalStudyAssistant) -> tuple[str, str | None]:
    with st.sidebar:
        st.title("📚 Study Assistant")
        st.caption("Ask questions and get answers from your own notes.")

        st.subheader("Add study materials")
        st.caption("Upload a file, index it once, then ask questions in the chat.")
        uploads = st.file_uploader(
            "PDF, TXT, Markdown, or DOCX",
            type=["pdf", "txt", "md", "markdown", "docx"],
            accept_multiple_files=True,
            help="Maximum 25 MB per file. Files are processed locally.",
        )
        latest_indexed_document_id: str | None = None
        if st.button("Index selected files", disabled=not uploads, use_container_width=True):
            progress = st.progress(0, text="Preparing documents…")
            for index, uploaded in enumerate(uploads or []):
                try:
                    document = assistant.ingest(uploaded.name, uploaded.getvalue())
                    st.success(f"Indexed {document['name']} ({document['chunks_count']} chunks)")
                    st.session_state["chat_mode"] = "Use my documents"
                    latest_indexed_document_id = document["id"]
                except AssistantError as error:
                    st.error(f"{uploaded.name}: {error}")
            progress.progress((index + 1) / len(uploads), text="Processing documents…")
            progress.empty()
            if latest_indexed_document_id:
                st.session_state["preferred_document_scope"] = latest_indexed_document_id
                st.session_state.pop("document_scope", None)
                st.rerun()

        st.subheader("Your documents")
        documents = repository.list_documents()
        indexed = [item for item in documents if item["status"] == "indexed"]
        if not documents:
            st.caption("No documents yet. Upload notes above to get started.")
        for document in documents:
            label = f"{'✅' if document['status'] == 'indexed' else '⚠️'} {document['name']}"
            with st.expander(label):
                st.caption(f"Status: {document['status']}")
                if document["status"] == "indexed":
                    st.caption(f"{document['chunks_count']} searchable chunks")
                elif document["error"]:
                    st.error(document["error"])
                if st.button("Delete document", key=f"delete-{document['id']}"):
                    assistant.delete_document(document["id"])
                    if st.session_state.get("document_scope") == document["id"]:
                        st.session_state.pop("document_scope", None)
                    if st.session_state.get("preferred_document_scope") == document["id"]:
                        st.session_state.pop("preferred_document_scope", None)
                    st.rerun()

        st.subheader("Study sessions")
        sessions = repository.list_sessions()
        if st.button("＋ New conversation", use_container_width=True):
            new_id = repository.create_session()
            st.session_state["active_session_id"] = new_id
            st.rerun()

        if not sessions:
            session_id = repository.create_session()
            st.session_state["active_session_id"] = session_id
            sessions = repository.list_sessions()

        active_id = st.session_state.get("active_session_id")
        session_by_id = {session["id"]: session for session in sessions}
        if active_id not in session_by_id:
            active_id = sessions[0]["id"]
            st.session_state["active_session_id"] = active_id
        selected_index = next(
            index for index, session in enumerate(sessions) if session["id"] == active_id
        )
        selected_session_id = st.selectbox(
            "Previous conversations",
            options=[session["id"] for session in sessions],
            format_func=lambda session_id: session_by_id[session_id]["title"],
            index=selected_index,
            label_visibility="collapsed",
        )
        st.session_state["active_session_id"] = selected_session_id

        scope_options = {
            None: "All indexed documents",
            **{
                document["id"]: f"{document['name']} · {document['id'][:8]}"
                for document in indexed
            },
        }
        scope_ids = list(scope_options)
        preferred_scope = st.session_state.get("preferred_document_scope")
        if preferred_scope not in scope_options:
            preferred_scope = None
            st.session_state.pop("preferred_document_scope", None)
        scope_index = scope_ids.index(preferred_scope)
        selected_document_id = st.selectbox(
            "Search in",
            options=scope_ids,
            format_func=lambda document_id: scope_options[document_id],
            index=scope_index,
            key="document_scope",
        )

        st.divider()
        st.caption(f"Model: `{get_settings().llm_model}`")
        st.caption(f"Embeddings: `{get_settings().embedding_model}`")

    return selected_session_id, selected_document_id


def render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.text(source["citation"])
            st.code(source["excerpt"], language=None)


def main() -> None:
    st.set_page_config(
        page_title="AI College Study Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("AI College Study Assistant")
    st.write(
        "Chat with your local AI study assistant. Ask general study questions "
        "or ground answers in your uploaded notes."
    )

    repository, assistant = get_services()
    session_id, document_id = render_sidebar(repository, assistant)
    indexed_documents = repository.list_documents(status="indexed")
    if indexed_documents:
        chat_mode = st.radio(
            "Answer mode",
            ["Use my documents", "General study help"],
            horizontal=True,
            key="chat_mode",
            help=(
                "Document mode answers from retrieved notes and cites sources. "
                "General mode uses the local model without document citations."
            ),
        )
        if chat_mode == "Use my documents":
            selected_scope = "all indexed documents"
            if document_id:
                selected_scope = next(
                    (
                        document["name"]
                        for document in indexed_documents
                        if document["id"] == document_id
                    ),
                    "the selected document",
                )
            st.info(
                f"Document chat is on. I’ll search {selected_scope} and show "
                "the source passages used in the answer."
            )
        else:
            st.info(
                "General study help is on. Switch to **Use my documents** to "
                "search your uploaded notes and see citations."
            )
    else:
        chat_mode = "General study help"
        st.info(
            "To chat with your own document: upload a PDF, TXT, Markdown, or "
            "DOCX file in the sidebar and click **Index selected files**. "
            "Then ask your question here; the assistant will retrieve relevant "
            "passages and cite them."
        )

    try:
        messages = repository.list_messages(session_id)
    except AssistantError as error:
        st.error(f"Could not load this conversation: {error}")
        return

    if not messages:
        st.markdown(
            "#### Ready when you are\n"
            "Ask for an explanation, examples, an analogy, or help reviewing "
            "a difficult concept."
        )

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message["sources"])

    question = st.chat_input(
        "Ask your study assistant anything…",
    )
    if question:
        repository.add_message(session_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching your notes and preparing an answer…"
                if chat_mode == "Use my documents"
                else "Thinking through your question…"
            ):
                try:
                    if chat_mode == "Use my documents":
                        answer, sources = assistant.answer(
                            question,
                            document_id=document_id,
                            history=messages,
                        )
                    else:
                        answer = assistant.general_answer(question, history=messages)
                        sources = []
                    st.markdown(answer)
                    render_sources(sources)
                    repository.add_message(
                        session_id,
                        "assistant",
                        answer,
                        sources=sources,
                    )
                    repository.update_session_title(session_id, question)
                except AssistantError as error:
                    st.error(str(error))


if __name__ == "__main__":
    main()
