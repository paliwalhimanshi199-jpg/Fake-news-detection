"""SQLite store for articles added by the user, with their predicted label."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "news.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL,
                score REAL,
                created_at TEXT NOT NULL
            )
            """
        )


def add_article(title: str, body: str, source: str, label: str, score: float | None) -> None:
    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO articles (title, body, source, label, score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (title, body, source, label, score, created),
        )


def list_articles(label: str = "All", keyword: str = "", limit: int = 500) -> list[dict]:
    sql = "SELECT id, title, body, source, label, score, created_at FROM articles WHERE 1 = 1"
    params: list = []
    if label and label != "All":
        sql += " AND label = ?"
        params.append(label)
    if keyword:
        sql += " AND (title LIKE ? OR body LIKE ? OR source LIKE ?)"
        pattern = f"%{keyword}%"
        params.extend([pattern, pattern, pattern])
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def counts() -> dict[str, int]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT label, COUNT(*) AS n FROM articles GROUP BY label").fetchall()
    return {row["label"]: row["n"] for row in rows}
