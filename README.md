# Multi-PDF RAG App with Sources

A Streamlit app that lets you upload several PDFs and ask questions about them. Each answer cites the file and page it came from. If the documents disagree, the app says so instead of picking one.

Built with LangChain, FAISS and OpenAI (`gpt-4o-mini` for answers, OpenAI embeddings for search).

## Features

- **Multiple PDFs:** upload any number of PDFs and they are searched as one collection.
- **Inline citations:** answers cite numbered context blocks like `[1][2]`, and the app lists only the cited sources as `file.pdf - page N`.
- **Conflict detection:** when sources contradict each other, the answer is shown in a "The documents disagree" box with what each source says.
- **No made-up answers:** if the answer isn't in the documents, the app replies "not found in the documents".
- **Adjustable retrieval:** sidebar sliders for chunk size, chunk overlap and the number of chunks retrieved (k).
- **Retrieved chunks viewer:** an expander shows the raw text of each retrieved chunk so you can check the answer.
- **Indexes only when needed:** the index is rebuilt only when the uploaded files or chunk settings change, not on every question.

## Screenshots

**PDFs uploaded and indexed**

![Upload and index](<Multi-PDF RAG  (1).png>)

**Answer with a conflict warning and sources**

![Answer with sources](<Multi-PDF RAG  (2).png>)

**Retrieved chunks with file and page labels**

![Retrieved chunks](<Multi-PDF RAG  (3).png>)

## Setup

Requires Python 3.9 or newer and an OpenAI API key.

1. Install the dependencies:

   ```bash
   pip install streamlit python-dotenv langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf
   ```

2. Create a `.env` file in this folder with your key:

   ```
   OPENAI_API_KEY=sk-...
   ```

   You can also set `OPENAI_API_KEY` as an environment variable in your terminal instead.

3. Run the app:

   ```bash
   streamlit run rag_app.py
   ```

   It opens at http://localhost:8501.

## How to use it

1. Upload one or more PDFs. The app splits them into chunks and shows how many were indexed.
2. Type a question in the text box.
3. Read the answer, the cited sources below it, and open **Retrieved chunks** to see the text the answer was based on.
4. If answers cite many scattered pages, try larger chunks or a lower k in the sidebar. Changing chunk settings re-indexes the files automatically.

## How it works

1. **Load:** each PDF is read page by page with `PyPDFLoader`. The real file name and page number are kept on every page.
2. **Split:** `RecursiveCharacterTextSplitter` breaks pages into overlapping chunks. Each chunk keeps its file and page.
3. **Embed and index:** chunks are embedded with `OpenAIEmbeddings` and stored in an in-memory FAISS index.
4. **Retrieve:** the top k chunks most similar to the question are fetched.
5. **Answer:** the chunks are numbered and labeled (`[1] Policy.pdf - page 80`) and sent to `gpt-4o-mini` with rules to cite blocks, flag conflicts and say "not found" when needed.
6. **Show sources:** the app reads the `[n]` citations from the answer and lists only those pages. If the model didn't cite anything, it lists all retrieved pages.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chat model | `CHAT_MODEL` in [rag_app.py](rag_app.py) | `gpt-4o-mini` |
| Chunk size | Sidebar | 1000 characters |
| Chunk overlap | Sidebar | 150 characters |
| Chunks to retrieve (k) | Sidebar | 4 |

## Limitations

- Scanned PDFs (images with no text layer) can't be read. Run OCR on them first.
- The index lives in memory, so it is rebuilt each time the app restarts.
- Conflict detection relies on the model. It can sometimes flag two sources as disagreeing when they only cover different parts of the question.

## Project files

```
rag_app.py               Streamlit app
Multi-PDF RAG  (1-3).png Screenshots
README.md                This file
```
