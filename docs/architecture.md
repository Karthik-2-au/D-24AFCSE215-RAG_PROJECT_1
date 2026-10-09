# Architecture and RAG workflow

## Chosen architecture

This project uses **Streamlit with a modular Python service layer**. It is a
good fit for an intermediate-level academic project because one Python
application can provide the interface and call clearly separated services,
without requiring a separate JavaScript frontend or API server to run locally.

All model inference is planned to use **Ollama** on the student's computer.
**SQLite** will store document and session metadata. **ChromaDB** will store
document embeddings and support similarity search. Uploaded content and
generated indexes are local application data, not source files.

## Architecture diagram

```text
┌──────────────────────────── Streamlit UI ───────────────────────────┐
│ Dashboard │ Documents │ Study Chat │ Practice │ History │ Settings  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │ Python application services│
                 │ ingestion / retrieval / RAG│
                 │ chat / quiz / flashcards   │
                 └───────┬────────────┬───────┘
                         │            │
               ┌─────────▼────┐  ┌────▼───────────┐
               │ SQLite       │  │ ChromaDB       │
               │ metadata and │  │ chunk vectors  │
               │ chat history │  │ and metadata   │
               └──────────────┘  └────┬───────────┘
                                       │
                              ┌────────▼────────┐
                              │ Ollama          │
                              │ local embeddings│
                              │ and local LLM   │
                              └─────────────────┘
```

The current app implements the Streamlit document/chat interface, local
ingestion and retrieval, persistent storage, and Ollama calls. Practice tools
and the full dashboard remain future work.

## How a question travels through RAG

1. The student selects general study chat or document-grounded chat.
2. In document mode, the student asks a question, optionally scoped to one
   document.
3. The application embeds the question using the configured local embedding
   model.
4. ChromaDB searches for the most relevant stored chunks, applying the
   document filter when one is selected.
5. The retrieval service checks whether results are relevant enough to use.
6. The RAG service constructs a prompt from the question and retrieved
   excerpts. Uploaded text is treated as untrusted evidence, never as
   instructions that override application rules.
7. Ollama generates a concise answer from that evidence. In general study
   mode, it answers with the language model without searching documents.
8. The UI shows citations linked to the retrieved filenames and page or chunk
   metadata. If evidence is insufficient, the response says so.
9. When history is enabled, the conversation is persisted in SQLite.

General study answers are not represented as facts from the user's documents.
The model should not be asked to reveal private chain-of-thought. Document
mode returns an answer and its supporting evidence instead.

## Module responsibilities

- `app.py`: Streamlit entry point and navigation.
- `config/settings.py`: validated environment-backed configuration.
- `backend/rag.py`: document parsing, chunking, Ollama embeddings, ChromaDB
  retrieval, and grounded answers.
- `database/repository.py`: parameterized SQLite operations for documents,
  study sessions, and messages.
- `prompts/rag_prompt.txt`: grounding, untrusted-document, and citation rules.
- `data/uploads/`: user-uploaded source files (never committed).
- `data/chroma/`: persistent vector index (never committed).
- `tests/`: parser, ingestion, retrieval, citations, persistence, and API/UI
  service tests as those components are implemented.

Study tools and diagnostics can be added as separate modules when implemented.

## Data flow and safeguards

Document uploads are validated by extension and size, then saved under a
generated UUID filename. Parsers extract text and page metadata where available.
Text is split into overlapping chunks, embedded locally, and stored with a
document ID, filename, page or section, and chunk index. A matching SQLite
record will track processing state and errors.

Deleting a document removes its database metadata, stored upload, and
associated vectors. Re-ingestion avoids accidental duplicate vectors.
Scanned PDFs with no extractable text should produce an actionable OCR
message; extracted file content is never executed.

SQLite statements will be parameterized. File paths will not be exposed to
chat users, errors should avoid disclosing document contents, and the chatbot
will not provide unrestricted filesystem or server access. Local processing
does not by itself guarantee complete privacy or security.

## Key configuration trade-offs

- **Chunk size:** small chunks are precise but can lose context; larger chunks
  retain context but may retrieve unrelated text.
- **Overlap:** helps keep facts that cross chunk boundaries together, at the
  cost of additional stored vectors.
- **`top_k`:** more chunks add context but can introduce noise and increase
  prompt length.
- **Model selection:** larger models can improve response quality but require
  more memory and may respond more slowly. The configured model names can be
  changed in the local `.env` file.
