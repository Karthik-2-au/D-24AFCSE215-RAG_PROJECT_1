"""Local document ingestion, retrieval, and evidence-grounded answer generation."""

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

import chromadb
import requests
from docx import Document as WordDocument
from pypdf import PdfReader

from config.settings import Settings
from backend.errors import AssistantError
from database.repository import StudyRepository

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}
INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough relevant evidence in the selected study materials "
    "to answer that reliably. Try another document or add notes that cover the topic."
)
RAG_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_prompt.txt"
GENERAL_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "general_prompt.txt"
)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split extracted text into bounded chunks with configurable overlap."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


class LocalStudyAssistant:
    """Use Ollama and ChromaDB to answer questions from indexed local files."""

    def __init__(self, settings: Settings, repository: StudyRepository) -> None:
        self.settings = settings
        self.repository = repository
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        try:
            self.vector_client = chromadb.PersistentClient(
                path=str(settings.data_dir / "chroma")
            )
            self.collection = self.vector_client.get_or_create_collection(
                name="study_materials",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise AssistantError(f"Could not open the local vector index: {error}") from error

    def _ollama_post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url.rstrip('/')}{endpoint}",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
        except requests.ConnectionError as error:
            raise AssistantError(
                "Ollama is not reachable. Start the Ollama app and try again."
            ) from error
        except requests.Timeout as error:
            raise AssistantError(
                "Ollama took too long to respond. Try a smaller local model or ask again."
            ) from error
        except requests.HTTPError as error:
            detail = ""
            if error.response is not None:
                detail = error.response.text[:300]
            raise AssistantError(
                f"Ollama request failed ({error.response.status_code if error.response else 'HTTP error'}). "
                f"Check that the configured model is installed. {detail}"
            ) from error
        except (requests.RequestException, ValueError) as error:
            raise AssistantError(f"Could not read a response from Ollama: {error}") from error
        if not isinstance(result, dict):
            raise AssistantError("Ollama returned an unexpected response.")
        return result

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 32):
            batch = texts[offset : offset + 32]
            result = self._ollama_post(
                "/api/embed",
                {"model": self.settings.embedding_model, "input": batch},
            )
            embeddings = result.get("embeddings")
            if (
                not isinstance(embeddings, list)
                or len(embeddings) != len(batch)
                or any(not isinstance(vector, list) for vector in embeddings)
            ):
                raise AssistantError(
                    "The embedding model returned invalid vectors. Check the configured embedding model."
                )
            vectors.extend(embeddings)
        return vectors

    def _extract(self, file_path: Path) -> list[tuple[int | None, str]]:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                reader = PdfReader(str(file_path))
                pages = [
                    (page_number, page.extract_text() or "")
                    for page_number, page in enumerate(reader.pages, start=1)
                ]
                if not any(text.strip() for _, text in pages):
                    raise AssistantError(
                        "No selectable text was found in this PDF. It may be scanned; OCR is needed."
                    )
                return pages
            if suffix == ".docx":
                document = WordDocument(str(file_path))
                return [(None, "\n".join(paragraph.text for paragraph in document.paragraphs))]
            if suffix in {".txt", ".md", ".markdown"}:
                return [(None, file_path.read_text(encoding="utf-8-sig"))]
        except AssistantError:
            raise
        except Exception as error:
            raise AssistantError(f"Could not extract text from this {suffix[1:].upper()} file: {error}") from error
        raise AssistantError("Unsupported file format. Upload PDF, TXT, Markdown, or DOCX.")

    def ingest(self, filename: str, content: bytes) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise AssistantError("Unsupported file format. Upload PDF, TXT, Markdown, or DOCX.")
        if not content:
            raise AssistantError("The uploaded file is empty.")
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise AssistantError(
                f"File is larger than the {self.settings.max_upload_mb} MB upload limit."
            )

        checksum = hashlib.sha256(content).hexdigest()
        duplicate = self.repository.find_document_by_checksum(checksum)
        if duplicate:
            raise AssistantError(
                f"This file is already indexed as “{duplicate['name']}”."
            )

        document_id = str(uuid.uuid4())
        safe_path = self.settings.data_dir / "uploads" / f"{document_id}{suffix}"
        display_name = Path(filename).name
        try:
            safe_path.write_bytes(content)
            self.repository.create_document(document_id, display_name, safe_path, checksum)
            extracted = self._extract(safe_path)
            texts: list[str] = []
            metadata: list[dict[str, Any]] = []
            for page_number, page_text in extracted:
                for chunk in split_text(
                    page_text,
                    self.settings.chunk_size,
                    self.settings.chunk_overlap,
                ):
                    chunk_index = len(texts)
                    texts.append(chunk)
                    metadata.append(
                        {
                            "document_id": document_id,
                            "filename": display_name,
                            "page": page_number or 0,
                            "chunk_index": chunk_index,
                        }
                    )
            if not texts:
                raise AssistantError("No readable text was found in this document.")
            embeddings = self._embed(texts)
            self.collection.add(
                ids=[f"{document_id}:{index}" for index in range(len(texts))],
                documents=texts,
                embeddings=embeddings,
                metadatas=metadata,
            )
            self.repository.finish_document(document_id, len(texts))
        except AssistantError as error:
            self._cleanup_failed_ingestion(document_id, safe_path, str(error))
            raise
        except Exception as error:
            message = f"Document indexing failed: {error}"
            self._cleanup_failed_ingestion(document_id, safe_path, message)
            raise AssistantError(message) from error

        document = self.repository.get_document(document_id)
        if not document:
            raise AssistantError("Document was indexed, but its metadata could not be loaded.")
        return document

    def _cleanup_failed_ingestion(
        self,
        document_id: str,
        file_path: Path,
        error_message: str,
    ) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
            file_path.unlink(missing_ok=True)
            if self.repository.get_document(document_id):
                self.repository.fail_document(document_id, error_message)
        except Exception as error:
            raise AssistantError(
                f"{error_message} Cleanup also failed: {error}"
            ) from error

    def delete_document(self, document_id: str) -> None:
        document = self.repository.get_document(document_id)
        if not document:
            raise AssistantError("That document no longer exists.")
        try:
            self.collection.delete(where={"document_id": document_id})
            path = Path(document["file_path"]).resolve()
            upload_root = (self.settings.data_dir / "uploads").resolve()
            if path.parent != upload_root:
                raise AssistantError("Refusing to delete a file outside the upload folder.")
            path.unlink(missing_ok=True)
            self.repository.delete_document(document_id)
        except AssistantError:
            raise
        except Exception as error:
            raise AssistantError(f"Could not delete this document: {error}") from error

    def answer(
        self,
        question: str,
        document_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        clean_question = question.strip()
        if not clean_question:
            raise AssistantError("Enter a question first.")
        if len(clean_question) > 4000:
            raise AssistantError("Please keep questions under 4,000 characters.")
        indexed_documents = self.repository.list_documents(status="indexed")
        if not indexed_documents:
            raise AssistantError("Upload and index study materials before asking questions.")
        if document_id and not any(doc["id"] == document_id for doc in indexed_documents):
            raise AssistantError("The selected document is not currently indexed.")

        recent_history = (history or [])[-6:]
        history_context = "\n".join(
            f"{message['role']}: {str(message['content'])[:1000]}"
            for message in recent_history
        )
        retrieval_query = f"{history_context}\nStudent question: {clean_question}".strip()
        query_embedding = self._embed([retrieval_query])[0]
        query: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(
                self.settings.top_k,
                max(1, self.collection.count()),
            ),
            "include": ["documents", "metadatas", "distances"],
        }
        if document_id:
            query["where"] = {"document_id": document_id}
        try:
            results = self.collection.query(**query)
        except Exception as error:
            raise AssistantError(f"Could not search the local study index: {error}") from error

        docs = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not docs or not distances or distances[0] > self.settings.max_retrieval_distance:
            return INSUFFICIENT_EVIDENCE, []

        sources: list[dict[str, str]] = []
        evidence: list[str] = []
        for index, (text, metadata, distance) in enumerate(zip(docs, metadatas, distances), start=1):
            if distance > self.settings.max_retrieval_distance:
                continue
            if not text or not metadata:
                continue
            citation = self._citation(metadata, index)
            sources.append(
                {
                    "citation": citation,
                    "excerpt": text,
                }
            )
            evidence.append(f"[{index}] {citation}\n{text}")
        if not evidence:
            return INSUFFICIENT_EVIDENCE, []

        try:
            system_prompt = RAG_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as error:
            raise AssistantError(f"Could not load the answer instructions: {error}") from error
        prompt = (
            f"Recent conversation (context only, not evidence):\n"
            f"{history_context or '(new conversation)'}\n\n"
            f"Retrieved evidence:\n{chr(10).join(evidence)}\n\n"
            f"Student question: {clean_question}"
        )
        response = self._ollama_post(
            "/api/chat",
            {
                "model": self.settings.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": self.settings.temperature},
            },
        )
        message = response.get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise AssistantError("The language model returned an empty answer. Please try again.")
        return answer.strip(), sources

    def general_answer(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Answer a general learning question without claiming it came from notes."""
        clean_question = question.strip()
        if not clean_question:
            raise AssistantError("Enter a question first.")
        if len(clean_question) > 4000:
            raise AssistantError("Please keep questions under 4,000 characters.")
        try:
            system_prompt = GENERAL_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as error:
            raise AssistantError(f"Could not load the chatbot instructions: {error}") from error

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for message in (history or [])[-6:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content[:2000]})
        messages.append({"role": "user", "content": clean_question})

        response = self._ollama_post(
            "/api/chat",
            {
                "model": self.settings.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.settings.temperature},
            },
        )
        message = response.get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise AssistantError("The language model returned an empty answer. Please try again.")
        return answer.strip()

    @staticmethod
    def _citation(metadata: dict[str, Any], number: int) -> str:
        filename = str(metadata.get("filename", "Study material"))
        page = metadata.get("page", 0)
        if page:
            return f"[{number}] {filename}, page {page}"
        chunk_index = int(metadata.get("chunk_index", 0)) + 1
        return f"[{number}] {filename}, chunk {chunk_index}"
