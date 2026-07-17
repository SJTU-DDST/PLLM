from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class Storage:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (
            Path.home() / ".local" / "share" / "pllm" / "events.sqlite3"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS replays (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_text TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    generated_tokens INTEGER NOT NULL DEFAULT 0,
                    paused_at_token INTEGER
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    experiment_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    label TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'live'
                );
                """
            )
            self._ensure_column(connection, "replays", "generated_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "replays", "paused_at_token", "INTEGER")

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def add_event(
        self,
        event_type: str,
        state: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events(created_at, event_type, state, reason, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    event_type,
                    state,
                    reason,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [self._event_row(row) for row in rows]

    def create_replay(self, request_payload: dict[str, Any], status: str) -> str:
        replay_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO replays(id, created_at, updated_at, status, request_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    replay_id,
                    now,
                    now,
                    status,
                    json.dumps(request_payload, ensure_ascii=False),
                ),
            )
        return replay_id

    def update_replay(
        self,
        replay_id: str,
        status: str,
        response_text: str = "",
        error: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE replays
                SET updated_at = ?, status = ?, response_text = ?, error = ?
                WHERE id = ?
                """,
                (time.time(), status, response_text, error, replay_id),
            )

    def update_replay_progress(
        self,
        replay_id: str,
        generated_tokens: int,
        response_text: str | None = None,
    ) -> None:
        with self._connect() as connection:
            if response_text is None:
                connection.execute(
                    """
                    UPDATE replays
                    SET updated_at = ?, generated_tokens = ?
                    WHERE id = ?
                    """,
                    (time.time(), max(0, int(generated_tokens)), replay_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE replays
                    SET updated_at = ?, generated_tokens = ?, response_text = ?
                    WHERE id = ?
                    """,
                    (
                        time.time(),
                        max(0, int(generated_tokens)),
                        response_text,
                        replay_id,
                    ),
                )

    def pause_running_replays(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE replays
                SET updated_at = ?, status = 'paused', paused_at_token = generated_tokens
                WHERE status = 'running'
                """,
                (time.time(),),
            )
            return cursor.rowcount

    def resume_paused_replays(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE replays
                SET updated_at = ?, status = 'running'
                WHERE status = 'paused'
                """,
                (time.time(),),
            )
            return cursor.rowcount

    def add_experiment(
        self,
        experiment_type: str,
        label: str,
        metrics: dict[str, Any],
        status: str = "completed",
        source: str = "live",
    ) -> str:
        experiment_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments(
                    id, created_at, experiment_type, status, label, metrics_json, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    time.time(),
                    experiment_type,
                    status,
                    label,
                    json.dumps(metrics, ensure_ascii=False),
                    source,
                ),
            )
        return experiment_id

    def list_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result

    def get_replay(self, replay_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM replays WHERE id = ?", (replay_id,)
            ).fetchone()
        return self._replay_row(row) if row else None

    def list_replays(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM replays ORDER BY updated_at DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [self._replay_row(row) for row in rows]

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    @staticmethod
    def _replay_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        return result
