"""SQLite persistence for documents, study sessions, and chat messages."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.errors import AssistantError


class StudyRepository:
    """Store application metadata in a local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        status TEXT NOT NULL,
                        chunks_count INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS documents_checksum_idx
                        ON documents(checksum);
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        sources_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL
                    );
                    """
                )
        except sqlite3.Error as error:
            raise AssistantError(f"Could not initialize the local database: {error}") from error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def find_document_by_checksum(self, checksum: str) -> dict[str, Any] | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM documents WHERE checksum = ? AND status = 'indexed' LIMIT 1",
                    (checksum,),
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as error:
            raise AssistantError(f"Could not check for duplicate documents: {error}") from error

    def create_document(
        self,
        document_id: str,
        name: str,
        file_path: Path,
        checksum: str,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO documents
                       (id, name, file_path, checksum, status, created_at)
                       VALUES (?, ?, ?, ?, 'processing', ?)""",
                    (document_id, name, str(file_path), checksum, self._now()),
                )
        except sqlite3.Error as error:
            raise AssistantError(f"Could not register the uploaded document: {error}") from error

    def finish_document(self, document_id: str, chunks_count: int) -> None:
        self._update_document(
            "UPDATE documents SET status = 'indexed', chunks_count = ?, error = NULL WHERE id = ?",
            (chunks_count, document_id),
        )

    def fail_document(self, document_id: str, error_message: str) -> None:
        self._update_document(
            "UPDATE documents SET status = 'failed', error = ? WHERE id = ?",
            (error_message, document_id),
        )

    def _update_document(self, query: str, values: tuple[Any, ...]) -> None:
        try:
            with self._connect() as connection:
                connection.execute(query, values)
        except sqlite3.Error as error:
            raise AssistantError(f"Could not update document processing status: {error}") from error

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                if status:
                    rows = connection.execute(
                        "SELECT * FROM documents WHERE status = ? ORDER BY created_at DESC",
                        (status,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT * FROM documents ORDER BY created_at DESC"
                    ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as error:
            raise AssistantError(f"Could not list documents: {error}") from error

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as error:
            raise AssistantError(f"Could not find the selected document: {error}") from error

    def delete_document(self, document_id: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        except sqlite3.Error as error:
            raise AssistantError(f"Could not remove document metadata: {error}") from error

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        now = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, "New conversation", now, now),
                )
            return session_id
        except sqlite3.Error as error:
            raise AssistantError(f"Could not create a study session: {error}") from error

    def list_sessions(self) -> list[dict[str, str]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM sessions ORDER BY updated_at DESC"
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as error:
            raise AssistantError(f"Could not load study sessions: {error}") from error

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT role, content, sources_json, created_at FROM messages
                       WHERE session_id = ? ORDER BY created_at, rowid""",
                    (session_id,),
                ).fetchall()
                return [
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "sources": json.loads(row["sources_json"]),
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]
        except (sqlite3.Error, json.JSONDecodeError) as error:
            raise AssistantError(f"Could not load conversation messages: {error}") from error

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, str]] | None = None,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO messages
                       (id, session_id, role, content, sources_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        session_id,
                        role,
                        content,
                        json.dumps(sources or [], ensure_ascii=False),
                        self._now(),
                    ),
                )
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (self._now(), session_id),
                )
        except sqlite3.Error as error:
            raise AssistantError(f"Could not save this message: {error}") from error

    def update_session_title(self, session_id: str, first_question: str) -> None:
        title = " ".join(first_question.split())[:60] or "New conversation"
        try:
            with self._connect() as connection:
                connection.execute(
                    """UPDATE sessions SET title = ?, updated_at = ?
                       WHERE id = ? AND title = 'New conversation'""",
                    (title, self._now(), session_id),
                )
        except sqlite3.Error as error:
            raise AssistantError(f"Could not update the study session title: {error}") from error
