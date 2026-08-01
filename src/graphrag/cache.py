"""危险货物问答系统的三级缓存实现。

L1：短期精确问答缓存，优先级最高。
L2：持久化语义答案缓存，用于匹配同一问题的不同问法。
L3：GB 12268 实体缓存，用于减少对 Neo4j 的重复查询。
"""

import json
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from cachetools import TTLCache

from .definitions import COMMON_DG_NAMES, DOMAIN_ENTITIES
from .query_understanding import analyze_query
from .settings import settings


# ==================== 公共：问题规范化与语义约束 ====================
def normalize_question(question: str) -> str:
    """生成稳定的精确缓存键，统一大小写、全半角、空白和句末标点。"""
    text = unicodedata.normalize("NFKC", question).lower().strip()
    text = re.sub(r"\s+", "", text)
    return text.rstrip("?!。！？")


def detect_intent(question: str) -> str:
    """提取粗粒度问题意图，用于防止语义缓存错误复用。"""
    structured = re.search(r"(?:^|;\s*)target=([^;]+)", question)
    if structured:
        return structured.group(1).strip()
    return analyze_query(question).target


def extract_cache_entity(question: str) -> str:
    """抽取用于语义缓存校验的实体；宁可少命中，也不跨实体复用答案。"""
    structured = re.search(r"(?:^|;\s*)subject=([^;]+)", question)
    if structured:
        entity = structured.group(1).strip()
        if entity and entity != "无主体":
            return entity
    analysis = analyze_query(question)
    if analysis.subject:
        return analysis.subject
    for entity in COMMON_DG_NAMES + DOMAIN_ENTITIES:
        if entity in question:
            return entity
    return ""


# ==================== L1：短期精确问答缓存 ====================
class L1AnswerCache:
    """线程安全的 TTL + LRU 缓存，保存最近使用的完整问答结果。"""

    def __init__(self, maxsize: int = 20, ttl: int = 1200):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            return dict(value)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = dict(value)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._cache.maxsize,
                "ttl_seconds": int(self._cache.ttl),
                "hits": self.hits,
                "misses": self.misses,
            }


# ==================== L2：持久化语义答案缓存 ====================
@dataclass
class SemanticCandidate:
    """L2检索出的最佳候选，尚未代表最终缓存命中。"""

    cache_id: int
    response: Dict[str, Any]
    similarity: float
    matched_question: str


class L2SemanticCache:
    """保存历史问答向量，并返回通过实体、意图和路由约束的语义候选。

    候选只代表相似度达到最低阈值；是否复用由 API 层调用 LLM 复核决定。
    """

    def __init__(
        self,
        database_path: Path,
        minimum_threshold: float = 0.85,
        direct_threshold: float = 1.0,
        max_entries: int = 300,
    ):
        if not 0 <= minimum_threshold <= direct_threshold <= 1:
            raise ValueError("语义缓存阈值必须满足 0 <= minimum <= direct <= 1")
        self.database_path = database_path
        self.minimum_threshold = minimum_threshold
        self.direct_threshold = direct_threshold
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.rejections = 0
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    route TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_signature
                ON semantic_answers (
                    data_version, prompt_version, model, top_k, intent, entity
                )
                """
            )

    @staticmethod
    def _normalized_vector(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 0 else value

    def find_candidate(
        self,
        question: str,
        route: str,
        top_k: int,
        model: str,
        data_version: str,
        prompt_version: str,
        embed: Callable[[str], np.ndarray],
        intent: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> Optional[SemanticCandidate]:
        intent = intent or detect_intent(question)
        entity = entity if entity is not None else extract_cache_entity(question)
        query_vector = self._normalized_vector(embed(question))

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, question, vector, dimension, response_json
                FROM semantic_answers
                WHERE data_version = ?
                  AND prompt_version = ?
                  AND model = ?
                  AND top_k = ?
                  AND intent = ?
                  AND entity = ?
                  AND route = ?
                """,
                (
                    data_version, prompt_version, model, top_k,
                    intent, entity, route,
                ),
            ).fetchall()

            best_row = None
            best_score = -1.0
            for row in rows:
                vector = np.frombuffer(row["vector"], dtype=np.float32)
                if len(vector) != row["dimension"] or len(vector) != len(query_vector):
                    continue
                score = float(np.dot(query_vector, vector))
                if score > best_score:
                    best_score = score
                    best_row = row

            if best_row is None or best_score < self.minimum_threshold:
                self.misses += 1
                return None

            return SemanticCandidate(
                cache_id=best_row["id"],
                response=json.loads(best_row["response_json"]),
                similarity=best_score,
                matched_question=best_row["question"],
            )

    def accept(self, candidate: SemanticCandidate) -> None:
        """仅在规则直返或 LLM 复核通过后，才记录为真正命中。"""
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE semantic_answers
                SET hit_count = hit_count + 1, last_accessed_at = ?
                WHERE id = ?
                """,
                (time.time(), candidate.cache_id),
            )
            self.hits += 1

    def reject(self) -> None:
        with self._lock:
            self.rejections += 1

    def put(
        self,
        question: str,
        route: str,
        response: Dict[str, Any],
        top_k: int,
        model: str,
        data_version: str,
        prompt_version: str,
        embed: Callable[[str], np.ndarray],
        intent: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> None:
        normalized = normalize_question(question)
        intent = intent or detect_intent(question)
        entity = entity if entity is not None else extract_cache_entity(question)
        vector = self._normalized_vector(embed(question))
        now = time.time()

        with self._lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM semantic_answers
                WHERE normalized_question = ?
                  AND data_version = ?
                  AND prompt_version = ?
                  AND model = ?
                  AND top_k = ?
                LIMIT 1
                """,
                (normalized, data_version, prompt_version, model, top_k),
            ).fetchone()

            values = (
                question, normalized, intent, entity, route,
                vector.tobytes(), len(vector),
                json.dumps(response, ensure_ascii=False),
                data_version, prompt_version, model, top_k,
                now, now,
            )
            if existing:
                connection.execute(
                    """
                    UPDATE semantic_answers
                    SET question = ?, normalized_question = ?, intent = ?,
                        entity = ?, route = ?, vector = ?, dimension = ?,
                        response_json = ?, data_version = ?, prompt_version = ?,
                        model = ?, top_k = ?, created_at = ?,
                        last_accessed_at = ?, hit_count = 0
                    WHERE id = ?
                    """,
                    values + (existing["id"],),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO semantic_answers (
                        question, normalized_question, intent, entity, route,
                        vector, dimension, response_json, data_version,
                        prompt_version, model, top_k, created_at,
                        last_accessed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

            connection.execute(
                """
                DELETE FROM semantic_answers
                WHERE id IN (
                    SELECT id FROM semantic_answers
                    ORDER BY last_accessed_at ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM semantic_answers) - ?)
                )
                """,
                (self.max_entries,),
            )

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM semantic_answers")

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            size = connection.execute(
                "SELECT COUNT(*) FROM semantic_answers"
            ).fetchone()[0]
        return {
            "size": size,
            "maxsize": self.max_entries,
            "minimum_threshold": self.minimum_threshold,
            "direct_threshold": self.direct_threshold,
            "hits": self.hits,
            "misses": self.misses,
            "rejections": self.rejections,
            "persistent": True,
        }


# ==================== 记忆缓存：用户俗称 ↔ 专业名称 ====================
class AliasMemoryCache:
    """持久化保存用户俗称到专业名称的映射，减少重复 LLM 规范化。"""

    def __init__(self, database_path: Path, max_entries: int = 500):
        self.database_path = database_path
        self.max_entries = max_entries
        self._lock = threading.RLock()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    @staticmethod
    def _normalize_alias(value: str) -> str:
        return normalize_question(value)

    def get(self, alias: str) -> Optional[str]:
        key = self._normalize_alias(alias)
        if not key:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT canonical_name FROM entity_aliases WHERE alias = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            connection.execute(
                """
                UPDATE entity_aliases
                SET hit_count = hit_count + 1, last_accessed_at = ?
                WHERE alias = ?
                """,
                (time.time(), key),
            )
            return row["canonical_name"]

    def put(
        self,
        alias: str,
        canonical_name: str,
        source: str = "llm",
        confidence: float = 0.88,
    ) -> None:
        key = self._normalize_alias(alias)
        canonical = canonical_name.strip()
        if not key or not canonical or key == self._normalize_alias(canonical):
            return
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO entity_aliases (
                    alias, canonical_name, source, confidence,
                    created_at, last_accessed_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(alias) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (key, canonical, source, confidence, now, now),
            )
            connection.execute(
                """
                DELETE FROM entity_aliases
                WHERE alias IN (
                    SELECT alias FROM entity_aliases
                    ORDER BY last_accessed_at ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM entity_aliases) - ?)
                )
                """,
                (self.max_entries,),
            )

    def aliases(self) -> Dict[str, str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT alias, canonical_name FROM entity_aliases"
            ).fetchall()
        return {row["alias"]: row["canonical_name"] for row in rows}

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM entity_aliases")

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            size = connection.execute(
                "SELECT COUNT(*) FROM entity_aliases"
            ).fetchone()[0]
        return {
            "size": size,
            "maxsize": self.max_entries,
            "persistent": True,
            "database": str(self.database_path),
        }


# ==================== L3：Neo4j 实体缓存 ====================
class L3EntityCache:
    """缓存 GB 12268 实体查询结果，减少对 Neo4j 的重复访问。

    缓存键由数据版本和 UN 编号（或实体名称）组成，缓存值为图谱查询结果。
    """

    def __init__(
        self,
        maxsize: int = 200,
        ttl: int = 86400,
        data_version: str = "gb-2025-v1",
    ):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()
        self.data_version = data_version
        self.hits = 0
        self.misses = 0

    def _key(self, entity: str) -> str:
        return f"{self.data_version}:{entity}"

    def get(self, entity: str) -> Optional[str]:
        with self._lock:
            value = self._cache.get(self._key(entity))
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            return value

    def put(self, entity: str, value: str) -> None:
        with self._lock:
            self._cache[self._key(entity)] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._cache.maxsize,
                "ttl_seconds": int(self._cache.ttl),
                "hits": self.hits,
                "misses": self.misses,
                "data_version": self.data_version,
            }


# L3由图谱查询模块共享；L1和L2在API服务中按运行配置创建。
ALIAS_MEMORY_CACHE = AliasMemoryCache(
    database_path=settings.cache.alias_memory_db,
    max_entries=settings.cache.alias_memory_size,
)

L3_ENTITY_CACHE = L3EntityCache(
    maxsize=settings.cache.l3_size,
    ttl=settings.cache.l3_ttl,
    data_version=settings.cache.data_version,
)
