import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Document AI | RAG", page_icon="📚", layout="wide")
API_URL = st.sidebar.text_input("FastAPI server", "http://127.0.0.1:8000").rstrip("/")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

st.title("📚 AI-Powered Document Question Answering")
st.caption("Upload a PDF and ask questions grounded in its content.")

with st.sidebar:
    st.subheader("1. Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Process PDF", type="primary", disabled=uploaded_file is None):
        with st.spinner("Extracting text and creating embeddings..."):
            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    data={"session_id": st.session_state.session_id},
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.uploaded = True
                st.session_state.history = []
                st.success(f"Indexed {result['chunks_indexed']} text chunks.")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")
    st.divider()
    st.caption("Your document is held in the API process memory for this demo. Restarting the API clears it.")
    if st.button("Clear document and chat"):
        try:
            requests.delete(f"{API_URL}/documents/{st.session_state.session_id}", timeout=10)
        except requests.RequestException:
            pass
        st.session_state.uploaded = False
        st.session_state.history = []
        st.rerun()

if not st.session_state.uploaded:
    st.info("Upload and process a PDF from the sidebar to begin.")
else:
    question = st.chat_input("Ask something about your document...")
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Retrieved source excerpts"):
                    for src in item["sources"]:
                        st.markdown(f"**Page {src['page']}**")
                        st.caption(src["excerpt"])
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"session_id": st.session_state.session_id, "question": question},
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.markdown(result["answer"])
                    with st.expander("Retrieved source excerpts"):
                        for src in result.get("sources", []):
                            st.markdown(f"**Page {src['page']}**")
                            st.caption(src["excerpt"])
                    st.session_state.history.append({
                        "role": "assistant", "content": result["answer"],
                        "sources": result.get("sources", [])
                    })
                except requests.RequestException as exc:
                    st.error(f"Could not get an answer: {exc}")
