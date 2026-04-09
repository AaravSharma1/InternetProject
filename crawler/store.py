"""
Content store — persists every fetched page and its outbound links to SQLite.

Schema
------
pages  : one row per URL (upsert on re-crawl)
links  : directed edge (src_url, dst_url) with anchor text & context

The store is opened once per crawl run and is NOT thread-safe on its own;
the crawler holds a single ContentStore and accesses it from the asyncio
event loop (single-threaded), so no locking is needed.
"""

import sqlite3
import time
from typing import List, Optional

from .parser import ExtractedLink


_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS pages (
    url         TEXT    PRIMARY KEY,
    fetch_time  REAL    NOT NULL,
    status_code INTEGER,
    byte_size   INTEGER NOT NULL DEFAULT 0,
    title       TEXT    NOT NULL DEFAULT '',
    html        TEXT    NOT NULL DEFAULT '',
    text        TEXT    NOT NULL DEFAULT '',
    error       TEXT
);

CREATE TABLE IF NOT EXISTS links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src_url     TEXT    NOT NULL,
    dst_url     TEXT    NOT NULL,
    anchor_text TEXT    NOT NULL DEFAULT '',
    context     TEXT    NOT NULL DEFAULT '',
    UNIQUE (src_url, dst_url)
);

CREATE INDEX IF NOT EXISTS idx_links_src ON links (src_url);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links (dst_url);
"""


class ContentStore:
    def __init__(self, db_path: str = "crawl.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def save_page(
        self,
        *,
        url: str,
        fetch_time: float,
        status_code: Optional[int],
        byte_size: int,
        title: str = "",
        html: str = "",
        text: str = "",
        error: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO pages
                (url, fetch_time, status_code, byte_size, title, html, text, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (url, fetch_time, status_code, byte_size, title, html, text, error),
        )
        self._conn.commit()

    def save_links(self, src_url: str, links: List[ExtractedLink]) -> None:
        if not links:
            return
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO links (src_url, dst_url, anchor_text, context)
            VALUES (?, ?, ?, ?)
            """,
            [(src_url, lnk.url, lnk.anchor_text, lnk.context) for lnk in links],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def page_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pages WHERE error IS NULL"
        ).fetchone()
        return row[0] if row else 0

    def link_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM links").fetchone()
        return row[0] if row else 0

    def get_page(self, url: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT url, fetch_time, status_code, byte_size, title, text, error "
            "FROM pages WHERE url = ?",
            (url,),
        ).fetchone()
        if row is None:
            return None
        keys = ("url", "fetch_time", "status_code", "byte_size", "title", "text", "error")
        return dict(zip(keys, row))

    def get_outlinks(self, src_url: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT dst_url, anchor_text, context FROM links WHERE src_url = ?",
            (src_url,),
        ).fetchall()
        return [{"dst_url": r[0], "anchor_text": r[1], "context": r[2]} for r in rows]

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
