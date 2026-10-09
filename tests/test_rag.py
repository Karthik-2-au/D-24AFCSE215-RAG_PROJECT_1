"""Focused tests for chunking and file type validation."""

import unittest
from types import SimpleNamespace
from typing import Any

from backend.rag import INSUFFICIENT_EVIDENCE, SUPPORTED_SUFFIXES, LocalStudyAssistant, split_text


class SplitTextTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(split_text(" \n\t ", chunk_size=100, overlap=10), [])

    def test_chunks_respect_size_and_preserve_overlap(self) -> None:
        text = " ".join(f"word{index}" for index in range(100))
        chunks = split_text(text, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertTrue(any(
            set(previous.split()) & set(current.split())
            for previous, current in zip(chunks, chunks[1:])
        ))

    def test_supported_formats_are_explicit(self) -> None:
        self.assertEqual(
            SUPPORTED_SUFFIXES,
            {".pdf", ".txt", ".md", ".markdown", ".docx"},
        )


class RetrievalTests(unittest.TestCase):
    def make_assistant(
        self,
        distance: float,
        chat_response: dict[str, Any] | None = None,
    ) -> LocalStudyAssistant:
        assistant = LocalStudyAssistant.__new__(LocalStudyAssistant)
        assistant.settings = SimpleNamespace(
            top_k=3,
            max_retrieval_distance=0.75,
            embedding_model="test-embedding",
            llm_model="test-chat",
            temperature=0.0,
        )
        assistant.repository = SimpleNamespace(
            list_documents=lambda status: [{"id": "doc-1", "status": status}]
        )
        assistant.query_arguments: dict[str, Any] = {}

        def fake_query(**kwargs: Any) -> dict[str, Any]:
            assistant.query_arguments.update(kwargs)
            return {
                "ids": [["doc-1:0"]],
                "documents": [["A stack follows last-in, first-out order."]],
                "metadatas": [[
                    {
                        "filename": "data-structures.pdf",
                        "page": 3,
                        "chunk_index": 0,
                    }
                ]],
                "distances": [[distance]],
            }

        assistant.collection = SimpleNamespace(count=lambda: 1, query=fake_query)
        assistant._embed = lambda texts: [[0.1, 0.2]]
        assistant._ollama_post = lambda endpoint, payload: (
            chat_response or {"message": {"content": "A stack is last-in, first-out [1]."}}
        )
        return assistant

    def test_low_relevance_evidence_is_not_sent_to_model(self) -> None:
        assistant = self.make_assistant(distance=1.1)

        answer, sources = assistant.answer("How does a stack work?")

        self.assertEqual(answer, INSUFFICIENT_EVIDENCE)
        self.assertEqual(sources, [])

    def test_answer_returns_metadata_for_a_citable_retrieved_source(self) -> None:
        assistant = self.make_assistant(distance=0.2)

        answer, sources = assistant.answer("How does a stack work?")

        self.assertIn("[1]", answer)
        self.assertEqual(sources[0]["citation"], "[1] data-structures.pdf, page 3")
        self.assertIn("last-in, first-out", sources[0]["excerpt"])

    def test_document_chat_limits_retrieval_to_the_selected_document(self) -> None:
        assistant = self.make_assistant(distance=0.2)

        assistant.answer("How does a stack work?", document_id="doc-1")

        self.assertEqual(
            assistant.query_arguments["where"],
            {"document_id": "doc-1"},
        )

    def test_general_chat_works_without_documents_or_retrieval(self) -> None:
        assistant = self.make_assistant(distance=1.1)
        assistant._ollama_post = lambda endpoint, payload: {
            "message": {"content": "A variable stores a value so a program can use it."}
        }

        answer = assistant.general_answer("What is a variable?")

        self.assertIn("stores a value", answer)

    def test_general_chat_sends_conversation_history_to_ollama(self) -> None:
        assistant = self.make_assistant(distance=1.1)
        captured: dict[str, Any] = {}

        def fake_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
            captured.update(payload)
            return {"message": {"content": "You can use a loop to repeat the task."}}

        assistant._ollama_post = fake_post
        answer = assistant.general_answer(
            "How do I repeat it?",
            history=[
                {"role": "user", "content": "How do I print five numbers?"},
                {"role": "assistant", "content": "You can print them one by one."},
            ],
        )

        self.assertIn("loop", answer)
        self.assertEqual(
            [message["role"] for message in captured["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            captured["messages"][1]["content"],
            "How do I print five numbers?",
        )


if __name__ == "__main__":
    unittest.main()
