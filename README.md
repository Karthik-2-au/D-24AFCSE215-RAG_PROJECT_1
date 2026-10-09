# AI College Study Assistant

A local-first study chatbot that can answer general study questions right
away, or search your course materials and answer with source citations. The
working MVP supports PDF, TXT, Markdown, and DOCX uploads; local Ollama
embeddings and chat; persistent ChromaDB search; and SQLite conversation
history.

## Architecture

```text
Streamlit interface
      │
      ├── SQLite: documents, study sessions, chat history
      ├── ChromaDB: locally persisted text chunks and embeddings
      └── Ollama: local embeddings + local language model
```

The app extracts text, splits it into overlapping chunks, and embeds those
chunks locally. When you ask a question, it retrieves relevant chunks and
asks Ollama to answer using that evidence. Citations are shown with excerpts.
If the selected materials do not provide relevant evidence, the assistant says
so instead of answering from unsupported guesses. Instructions embedded in
uploaded documents are treated as untrusted text.

## Requirements

- Windows 10 or 11, PowerShell, and Python 3.10 or newer.
- Ollama installed and running.
- Sufficient disk/memory for your selected Ollama models.

## Install and run

In PowerShell, from this project directory:

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama list
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal (normally
`http://localhost:8501`). General study chat is available immediately, even
before uploading documents. To get grounded answers, upload notes in the
sidebar, select **Index selected files**, and wait for indexing to finish. The
app automatically switches to **Use my documents** after an upload is indexed.
Ask a question in the chat, such as “What are the stages of the water cycle?”
The assistant retrieves relevant passages and shows citations and excerpts.
Choose **All indexed documents** or one document in the sidebar to scope the
search. If your notes do not contain a useful answer, document mode says so
instead of guessing. **General study help** is available as a separate mode
and does not use document evidence.

If PowerShell blocks virtual-environment activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell
window and retry. Alternatively, run commands without activation using
`.\.venv\Scripts\python.exe` and
`.\.venv\Scripts\streamlit.exe`.

The first installation may take several minutes while Python dependencies and
the Ollama model files download. Start Ollama before
starting an indexing or answer request. Use `ollama list` to see installed
models.

## Configuration

Copy `.env.example` to `.env` and edit settings prefixed with
`STUDY_ASSISTANT_`. For example, change `LLM_MODEL` to another model that
`ollama list` reports as installed. `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K`
control retrieval:

- Smaller chunks can be more precise but may split context.
- Overlap preserves facts spanning chunk boundaries but stores more vectors.
- A larger `TOP_K` gives the model more context but can add irrelevant text.
- `MAX_RETRIEVAL_DISTANCE` controls how far a ChromaDB match may be before the
  assistant treats the evidence as insufficient.

## Local data and privacy

Uploaded files, the ChromaDB index, and the SQLite database are stored under
`data/`, which is ignored by Git. Use the document delete action to remove an
upload and its vectors. Local inference avoids a paid model API, but local
processing does not by itself guarantee complete privacy or security.

## Verify

Run unit tests from PowerShell:

```powershell
python -m unittest discover -v
```

Supported uploads are PDF, TXT, Markdown, and DOCX files up to 25 MB. Text
inside scanned/image-only PDFs cannot be extracted without OCR, so those PDFs
need OCR before indexing.

The current tests cover text chunking, supported formats, low-relevance
evidence, and citation metadata. A local Ollama + Chroma smoke test also
indexed a sample note and returned a cited answer. Broader answer-quality
benchmarks, OCR, quiz tools, and evaluation metrics remain future improvements;
no benchmark results are claimed.

## Project structure

```text
app.py                 Streamlit chat and document UI
backend/errors.py      Shared user-displayable errors
backend/rag.py         Parsing, chunking, Ollama, ChromaDB, grounded answers
config/settings.py     Validated environment configuration
database/repository.py SQLite metadata and conversation history
prompts/rag_prompt.txt Grounding and citation instructions
data/                  Local uploads, database, and vector index (ignored)
tests/                 Focused unit tests
docs/architecture.md   Architecture and RAG explanation
```

Future work includes chapter summaries, quiz generation, flashcards, richer
session controls, diagnostics, and a larger evaluation suite.
