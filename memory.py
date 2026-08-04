import sqlite3
from datetime import UTC, datetime

from settings import MEMORY_DB_PATH


connection = sqlite3.connect(database=MEMORY_DB_PATH, check_same_thread=False)
connection.row_factory = sqlite3.Row


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def setup_thread_metadata() -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()


def list_threads() -> list[dict]:
    setup_thread_metadata()
    rows = connection.execute(
        """
        SELECT thread_id, title, created_at, updated_at
        FROM conversation_threads
        ORDER BY updated_at DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def save_thread(thread_id: str, title: str = "") -> None:
    setup_thread_metadata()
    timestamp = now_iso()
    connection.execute(
        """
        INSERT INTO conversation_threads (thread_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            title = CASE
                WHEN excluded.title != '' THEN excluded.title
                ELSE conversation_threads.title
            END,
            updated_at = excluded.updated_at
        """,
        (thread_id, title, timestamp, timestamp),
    )
    connection.commit()


def update_thread_title(thread_id: str, title: str) -> None:
    save_thread(thread_id, title)


def delete_thread(thread_id: str) -> None:
    setup_thread_metadata()

    for table_name in ("conversation_threads", "checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        try:
            connection.execute(f"DELETE FROM {table_name} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            continue

    connection.commit()
