"""Memory Manager — knowledge graph + rolling context for the living agent.

Two memory systems working together:
  1. Knowledge Graph (long-term): Entities, relationships, semantic search
     - SQLite + sqlite-vec (384-dim embeddings via fastembed)
     - Same architecture as claude-knowledge-graph plugin

  2. Rolling Context (short-term): Compressed conversation history
     - Content-hash based compression
     - Same architecture as claude-rolling-context plugin
     - Runs on base model (LoRA off) in background
"""

import json
import time
import sqlite3
import hashlib
import logging
import asyncio
from typing import Any, Optional
from pathlib import Path

log = logging.getLogger(__name__)


class KnowledgeGraph:
    """Persistent knowledge graph with semantic search.

    Stores entities (nodes) and relationships (edges) in SQLite.
    Semantic search via fastembed embeddings + sqlite-vec.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path.home() / ".agent" / "knowledge.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._embedder = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self):
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT DEFAULT '',
                properties TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(name, type)
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES entities(id),
                target_id INTEGER NOT NULL REFERENCES entities(id),
                type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                content_hash TEXT UNIQUE,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
        """)
        conn.commit()

    def store(self, entity: str, type: str, properties: dict,
              relations: list[dict] = None) -> dict:
        """Store or update an entity in the knowledge graph."""
        conn = self._get_conn()
        now = time.time()
        props_json = json.dumps(properties)

        # Upsert entity
        try:
            conn.execute(
                "INSERT INTO entities (name, type, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (entity, type, props_json, now, now),
            )
        except sqlite3.IntegrityError:
            conn.execute(
                "UPDATE entities SET properties = ?, updated_at = ? WHERE name = ? AND type = ?",
                (props_json, now, entity, type),
            )

        # Add relationships
        if relations:
            entity_id = conn.execute(
                "SELECT id FROM entities WHERE name = ? AND type = ?", (entity, type)
            ).fetchone()["id"]
            for rel in relations:
                target_row = conn.execute(
                    "SELECT id FROM entities WHERE name = ?", (rel["target"],)
                ).fetchone()
                if target_row:
                    conn.execute(
                        "INSERT INTO relationships (source_id, target_id, type, weight, created_at) VALUES (?, ?, ?, ?, ?)",
                        (entity_id, target_row["id"], rel.get("type", "related"), rel.get("weight", 1.0), now),
                    )

        conn.commit()
        return {"stored": entity, "type": type}

    def query(self, query: str, type: str = None, limit: int = 10) -> list[dict]:
        """Query the knowledge graph. Simple keyword search for now."""
        conn = self._get_conn()
        sql = "SELECT name, type, properties, updated_at FROM entities WHERE name LIKE ?"
        params = [f"%{query}%"]
        if type:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [
            {"entity": r["name"], "type": r["type"],
             "properties": json.loads(r["properties"]),
             "updated_at": r["updated_at"]}
            for r in rows
        ]

    def update(self, entity: str, properties: dict) -> dict:
        """Update an existing entity's properties (merges)."""
        conn = self._get_conn()
        row = conn.execute("SELECT id, properties FROM entities WHERE name = ?", (entity,)).fetchone()
        if not row:
            return {"error": f"Entity '{entity}' not found"}
        existing = json.loads(row["properties"])
        existing.update(properties)
        conn.execute(
            "UPDATE entities SET properties = ?, updated_at = ? WHERE id = ?",
            (json.dumps(existing), time.time(), row["id"]),
        )
        conn.commit()
        return {"updated": entity, "properties": existing}

    def traverse(self, entity: str, depth: int = 2) -> list[dict]:
        """BFS graph traversal from an entity."""
        conn = self._get_conn()
        start = conn.execute("SELECT id, name, type, properties FROM entities WHERE name = ?", (entity,)).fetchone()
        if not start:
            return []

        visited = {start["id"]}
        result = [{"name": start["name"], "type": start["type"],
                    "properties": json.loads(start["properties"]), "depth": 0}]
        frontier = [start["id"]]

        for d in range(1, depth + 1):
            next_frontier = []
            for node_id in frontier:
                # Both directions
                rels = conn.execute("""
                    SELECT r.type as rel_type, e.id, e.name, e.type, e.properties
                    FROM relationships r JOIN entities e ON e.id = r.target_id
                    WHERE r.source_id = ?
                    UNION
                    SELECT r.type as rel_type, e.id, e.name, e.type, e.properties
                    FROM relationships r JOIN entities e ON e.id = r.source_id
                    WHERE r.target_id = ?
                """, (node_id, node_id)).fetchall()
                for rel in rels:
                    if rel["id"] not in visited:
                        visited.add(rel["id"])
                        next_frontier.append(rel["id"])
                        result.append({
                            "name": rel["name"], "type": rel["type"],
                            "properties": json.loads(rel["properties"]),
                            "relation": rel["rel_type"], "depth": d,
                        })
            frontier = next_frontier

        return result


class RollingContext:
    """Rolling context compression — keeps recent context verbatim, compresses old.

    At runtime, this runs on the base model (LoRA off) in background.
    Uses content-hash based matching (same as claude-rolling-context proxy).
    """

    def __init__(self, db_path: str = None, max_tokens: int = 32000, target_tokens: int = 16000):
        self.db_path = db_path or str(Path.home() / ".agent" / "rolling_context.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS context_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                chunk_type TEXT NOT NULL,  -- 'raw' or 'compressed'
                content TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                original_tokens INTEGER NOT NULL,
                compressed_tokens INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ctx_conv ON context_chunks(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_comp_conv ON compressions(conversation_id);
        """)
        self._conn.commit()

    def store_turn(self, conversation_id: str, content: str):
        """Store a conversation turn for later compression."""
        conn = self._get_conn()
        token_est = len(content) // 4  # Rough estimate
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        conn.execute(
            "INSERT INTO context_chunks (conversation_id, chunk_type, content, token_estimate, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, "raw", content, token_est, content_hash, time.time()),
        )
        conn.commit()

    def get_context(self, conversation_id: str) -> list[dict]:
        """Get current context for a conversation (compressed + recent)."""
        conn = self._get_conn()
        # Get most recent compression
        comp = conn.execute(
            "SELECT summary FROM compressions WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        # Get raw chunks after last compression
        chunks = conn.execute(
            "SELECT content, created_at FROM context_chunks WHERE conversation_id = ? AND chunk_type = 'raw' ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()

        result = []
        if comp:
            result.append({"type": "summary", "content": comp["summary"]})
        for chunk in chunks:
            result.append({"type": "raw", "content": chunk["content"]})
        return result

    def needs_compression(self, conversation_id: str) -> bool:
        """Check if context exceeds max_tokens and needs compression."""
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) as total FROM context_chunks WHERE conversation_id = ? AND chunk_type = 'raw'",
            (conversation_id,),
        ).fetchone()["total"]
        return total > self.max_tokens

    async def compress(self, conversation_id: str, summarizer_fn) -> dict:
        """Compress old context using the base model (LoRA off).

        Args:
            conversation_id: Which conversation to compress
            summarizer_fn: async fn(text) -> summary  (runs on base model)
        """
        conn = self._get_conn()
        chunks = conn.execute(
            "SELECT id, content, token_estimate FROM context_chunks WHERE conversation_id = ? AND chunk_type = 'raw' ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()

        if not chunks:
            return {"compressed": 0}

        # Keep the most recent target_tokens worth of context
        total = sum(c["token_estimate"] for c in chunks)
        if total <= self.max_tokens:
            return {"compressed": 0}

        # Find split point: compress everything before the target window
        keep_tokens = 0
        split_idx = len(chunks)
        for i in range(len(chunks) - 1, -1, -1):
            keep_tokens += chunks[i]["token_estimate"]
            if keep_tokens >= self.target_tokens:
                split_idx = i
                break

        to_compress = chunks[:split_idx]
        if not to_compress:
            return {"compressed": 0}

        # Build text to compress
        text = "\n".join(c["content"] for c in to_compress)
        original_tokens = sum(c["token_estimate"] for c in to_compress)

        # Get existing summary to merge with
        existing = conn.execute(
            "SELECT summary FROM compressions WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if existing:
            text = f"[PREVIOUS SUMMARY]\n{existing['summary']}\n\n[NEW CONTENT TO MERGE]\n{text}"

        # Call summarizer (runs on base model in background)
        summary = await summarizer_fn(text)
        compressed_tokens = len(summary) // 4

        # Store compression
        conn.execute(
            "INSERT INTO compressions (conversation_id, summary, original_tokens, compressed_tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, summary, original_tokens, compressed_tokens, time.time()),
        )
        # Remove compressed raw chunks
        for chunk in to_compress:
            conn.execute("DELETE FROM context_chunks WHERE id = ?", (chunk["id"],))
        conn.commit()

        return {"compressed": original_tokens, "to": compressed_tokens}

    def recall(self, query: str, time_range: str = None) -> list[dict]:
        """Recall from compressed context (search summaries)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT summary, created_at FROM compressions WHERE summary LIKE ? ORDER BY created_at DESC LIMIT 5",
            (f"%{query}%",),
        ).fetchall()
        return [{"summary": r["summary"], "date": r["created_at"]} for r in rows]


class MemoryManager:
    """Unified interface for all memory operations.

    Dispatches tool calls to either KnowledgeGraph or RollingContext.
    """

    def __init__(self, workspace: str = None):
        base = Path(workspace or ".") / ".agent"
        self.graph = KnowledgeGraph(str(base / "knowledge.db"))
        self.context = RollingContext(str(base / "rolling_context.db"))

    async def execute(self, name: str, args: dict) -> str:
        """Execute a memory tool call."""
        try:
            if name == "memory_store":
                result = self.graph.store(**args)
            elif name == "memory_query":
                result = self.graph.query(**args)
            elif name == "memory_update":
                result = self.graph.update(**args)
            elif name == "context_compress":
                # This needs the summarizer function — handled by runtime
                return "Error: context_compress must be called through runtime"
            elif name == "context_recall":
                result = self.context.recall(**args)
            else:
                return f"Error: Unknown memory tool '{name}'"
            return json.dumps(result)
        except Exception as e:
            log.error(f"Memory tool {name} failed: {e}")
            return f"Error: {e}"

    @property
    def memory_tools(self) -> list[str]:
        return ["memory_store", "memory_query", "memory_update",
                "context_compress", "context_recall"]
