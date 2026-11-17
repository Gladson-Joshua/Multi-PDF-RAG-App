import os
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHAT_MODEL = "gpt-4o-mini"  # change to any chat model your key can use
NOT_FOUND = "not found in the documents"
CONFLICT_PREFIX = "⚠️ The documents disagree"

SYSTEM_PROMPT = f"""You are a careful assistant answering questions about the user's uploaded PDFs.
Use ONLY the numbered context blocks. Each block is labeled with its file and page.
Rules:
1. Cite the blocks you used inline, like [1] or [2][3].
2. If blocks from different files (or pages) give conflicting information, do NOT pick one.
   Start the answer with "{CONFLICT_PREFIX}:" and state what each source says, with its citation.
3. If the context does not contain the answer, reply exactly: "{NOT_FOUND}".
4. Be concise: 1-4 sentences unless the question asks for a list."""

# Load .env into the environment; the key is still read from an env variable.
load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG", page_icon=":material/library_books:")
st.title("Multi-PDF RAG app with sources")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to .env or your terminal, then restart.")
    st.stop()

# ---------- Settings ----------
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=25)
    top_k = st.slider("Chunks to retrieve (k)", 2, 10, 4)
    st.caption(
        "If answers regularly cite many different pages, chunks may be too "
        "small or k too high. Tune and re-index."
    )


def page_label(doc):
    """'file.pdf - page N' with 1-based page numbers (PyPDFLoader counts from 0)."""
    name = doc.metadata.get("source", "unknown file")
    page = doc.metadata.get("page")
    return f"{name} - page {page + 1}" if page is not None else name


def load_and_split(uploaded_file, splitter):
    """Save one upload to a temp file, load it page by page, and split it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    # PyPDFLoader sets "source" to the temp path; replace it with the real name.
    # "page" stays as-is and the splitter copies it into every chunk.
    for page in pages:
        page.metadata["source"] = uploaded_file.name

    return splitter.split_documents(pages)


# ---------- Upload & index ----------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if not files:
    for key in ("store", "index_key", "chunk_count"):
        st.session_state.pop(key, None)
    st.info("Upload one or more PDFs to build the index.")
    st.stop()

current_files = sorted((f.name, f.size) for f in files)
index_key = (tuple(current_files), chunk_size, chunk_overlap)

# Rebuild only when the files or chunk settings change, not on every rerun.
if st.session_state.get("index_key") != index_key:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    with st.spinner(f"Indexing {len(files)} file(s)..."):
        for f in files:
            try:
                chunks.extend(load_and_split(f, splitter))
            except Exception as exc:
                st.warning(f"Skipped {f.name}: {exc}")

        if not chunks:
            st.error("No text could be extracted. Scanned PDFs need OCR first.")
            st.stop()

        st.session_state.store = FAISS.from_documents(chunks, OpenAIEmbeddings())

    st.session_state.index_key = index_key
    st.session_state.chunk_count = len(chunks)

st.write("Uploaded: " + ", ".join(name for name, _ in current_files))
st.success(f"Indexed {st.session_state.chunk_count} chunks")

# ---------- Ask a question ----------
question = st.text_input("Ask a question about your documents")

if question.strip():
    with st.spinner("Searching the documents..."):
        retriever = st.session_state.store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)

        # Number and label each block so the model can cite it and spot conflicts.
        context = "\n\n---\n\n".join(
            f"[{i}] {page_label(d)}\n{d.page_content}" for i, d in enumerate(docs, 1)
        )
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        answer = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Context:\n{context}\n\nQuestion: {question}"),
            ]
        ).content.strip()

    st.markdown(f"**Q:** {question}")
    if answer.startswith(CONFLICT_PREFIX):
        body = answer[len(CONFLICT_PREFIX):].lstrip(" :")
        st.warning(body, title="The documents disagree")
    else:
        st.markdown(f"**A:** {answer}")

    if NOT_FOUND not in answer.lower():
        # Sources = only the blocks the answer actually cited, in order, de-duplicated.
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        cited_docs = [docs[n - 1] for n in dict.fromkeys(cited) if 1 <= n <= len(docs)]
        if not cited_docs:  # model forgot to cite: fall back to everything retrieved
            cited_docs = docs

        sources = list(dict.fromkeys(page_label(d) for d in cited_docs))
        st.caption("Sources: " + " | ".join(sources))
        st.caption(f"Answer drew on {len(sources)} distinct page(s).")

    with st.expander("Retrieved chunks"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**[{i}] {page_label(d)}**")
            st.text(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
