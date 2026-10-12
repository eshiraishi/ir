"""Record/replay cache for model calls.

Responses are stored in SQLite under ``sha256(canonical request JSON)``. Re-running any analysis
replays them byte-identically; this, not provider-side seeding, is TRICE's determinism guarantee
for hosted inference. S7 bypasses the cache and records repeated live calls in ``audit``.
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(backend: str, body: dict) -> str:
    """Key of one request. ``backend`` is the logical backend name (never a URL or port)."""
    return hashlib.sha256(canonical_json({"backend": backend, "body": body}).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL,"
            " created REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS audit ("
            " key TEXT NOT NULL, tag TEXT NOT NULL, response TEXT NOT NULL, created REAL NOT NULL,"
            " PRIMARY KEY (key, tag))"
        )

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, request: dict, response: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO responses VALUES (?, ?, ?, ?)",
                (key, canonical_json(request), canonical_json(response), time.time()),
            )

    def put_audit(self, key: str, tag: str, response: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO audit VALUES (?, ?, ?, ?)",
                (key, tag, canonical_json(response), time.time()),
            )

    def audit_responses(self, key: str) -> dict[str, dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT tag, response FROM audit WHERE key = ?", (key,)
            ).fetchall()
        return {tag: json.loads(resp) for tag, resp in rows}

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._db.close()
