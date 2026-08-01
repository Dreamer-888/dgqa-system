"""运行环境配置入口。

LLM 客户端配置仍保留在 llm 包内；本模块只集中管理检索、API、
缓存、Neo4j 和 Hugging Face 镜像等运行环境相关配置。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    """加载 .env，已存在的进程环境变量优先级更高。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def get_path(name: str, default: Path | str) -> Path:
    return Path(os.getenv(name, str(default)))


def get_csv(name: str, default: str) -> List[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class ApiSettings:
    host: str
    port: int
    reload: bool
    cors_allow_origins: List[str]
    rebuild_index: bool


@dataclass(frozen=True)
class CacheSettings:
    data_version: str
    prompt_version: str
    l1_size: int
    l1_ttl: int
    l2_db: Path
    l2_min_threshold: float
    l2_direct_threshold: float
    l2_size: int
    l2_enabled: bool
    l3_size: int
    l3_ttl: int
    alias_memory_db: Path
    alias_memory_size: int


@dataclass(frozen=True)
class ModelSettings:
    embedding_model: str
    rerank_model: str
    hf_endpoint: Optional[str]


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class QueryAnalysisSettings:
    llm_enabled: bool
    llm_threshold: float


@dataclass(frozen=True)
class AppSettings:
    api: ApiSettings
    cache: CacheSettings
    model: ModelSettings
    neo4j: Neo4jSettings
    query_analysis: QueryAnalysisSettings


def build_settings() -> AppSettings:
    load_env_file()
    l2_default = PROJECT_ROOT / "data" / "cache" / "semantic_cache.db"
    alias_default = PROJECT_ROOT / "data" / "cache" / "alias_memory.db"
    return AppSettings(
        api=ApiSettings(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=get_int("API_PORT", 8000),
            reload=get_bool("API_RELOAD", False),
            cors_allow_origins=get_csv("CORS_ALLOW_ORIGINS", "*"),
            rebuild_index=get_bool("REBUILD_INDEX", False),
        ),
        cache=CacheSettings(
            data_version=os.getenv("DATA_VERSION", "gb-2025-v1"),
            prompt_version=os.getenv("PROMPT_VERSION", "v1"),
            l1_size=get_int("L1_CACHE_SIZE", 20),
            l1_ttl=get_int("L1_CACHE_TTL", 1200),
            l2_db=get_path("L2_CACHE_DB", l2_default),
            l2_min_threshold=get_float("L2_CACHE_MIN_THRESHOLD", 0.85),
            l2_direct_threshold=get_float("L2_CACHE_DIRECT_THRESHOLD", 1.0),
            l2_size=get_int("L2_CACHE_SIZE", 5000),
            l2_enabled=get_bool("L2_CACHE_ENABLED", True),
            l3_size=get_int("L3_CACHE_SIZE", 200),
            l3_ttl=get_int("L3_CACHE_TTL", 86400),
            alias_memory_db=get_path("ALIAS_MEMORY_DB", alias_default),
            alias_memory_size=get_int("ALIAS_MEMORY_SIZE", 500),
        ),
        model=ModelSettings(
            embedding_model=os.getenv("EMBEDDING_MODEL", "./models/bge-m3"),
            rerank_model=os.getenv("RERANK_MODEL", "./models/bge-reranker-base"),
            hf_endpoint=os.getenv("HF_ENDPOINT") or None,
        ),
        neo4j=Neo4jSettings(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWD", "ubuntu123"),
        ),
        query_analysis=QueryAnalysisSettings(
            llm_enabled=get_bool("QUERY_ANALYSIS_LLM_ENABLED", True),
            llm_threshold=get_float("QUERY_ANALYSIS_LLM_THRESHOLD", 0.85),
        ),
    )


settings = build_settings()


def apply_runtime_environment() -> None:
    """应用确实需要写入进程环境的运行时配置。"""
    if settings.model.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.model.hf_endpoint)
