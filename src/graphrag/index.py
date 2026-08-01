"""索引与精确查表管理。

这个模块集中管理两类索引：
- 精确哈希索引：GB 12268 UN 条目、附录A特殊规定编号；
- 来源子索引：按 source_filter 缩小后的 FAISS/BM25 检索空间。

它不理解用户问题，也不构造最终证据格式。
"""

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .definitions import (
    GB12268_TABLE_PATH, SPECIAL_PROVISIONS_APPENDIX_PATH,
    SourceFilter,
)


@dataclass(frozen=True)
class SpecialProvisionLookup:
    """附录A特殊规定编号的精确查询结果。"""
    code: str
    found: bool
    source_name: str
    content: Optional[str] = None


@dataclass(frozen=True)
class TextSearchSpace:
    """一组互相对齐的文本检索索引。"""
    faiss_index: faiss.Index
    metadata: List[Dict]
    bm25_index: BM25Okapi


@dataclass(frozen=True)
class ClauseLookup:
    """metadata 中按章节号精确查询的结果。"""
    section_id: str
    found: bool
    chunks: List[Dict]


@lru_cache(maxsize=1)
def load_gb12268_un_numbers(path: Path = GB12268_TABLE_PATH) -> frozenset[str]:
    """从 GB 12268 品名表加载有效 UN 编号集合，用于裸数字 UN 校验。"""
    if not path.exists():
        return frozenset()

    numbers = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                raw = (row.get("un_number") or "").strip()
                digits = re.sub(r"\D", "", raw)
                if len(digits) <= 4 and digits:
                    numbers.add(f"UN{digits.zfill(4)}")
    except OSError:
        return frozenset()
    return frozenset(numbers)


@lru_cache(maxsize=1)
def load_gb12268_entity_names(path: Path = GB12268_TABLE_PATH) -> tuple[str, ...]:
    """从 GB 12268 品名表加载可用于 KG 查询的中文实体名。"""
    if not path.exists():
        return ()

    names = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = (row.get("name_zh") or "").strip()
                if name:
                    names.add(name)
    except OSError:
        return ()
    return tuple(sorted(names, key=len, reverse=True))


@lru_cache(maxsize=4)
def load_special_provisions_appendix(
    path: Path = SPECIAL_PROVISIONS_APPENDIX_PATH,) -> Dict[str, str]:
    """加载 GB 12268 附录A特殊规定文本，构造 编号 -> 正文 的哈希索引。"""
    if not path.exists():
        return {}

    entries: Dict[str, List[str]] = {}
    current_code: Optional[str] = None
    entry_pattern = re.compile(r"^\s*(\d{1,4})\s+(.+?)\s*$")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = entry_pattern.match(line)
        if match:
            current_code = match.group(1)
            entries.setdefault(current_code, []).append(match.group(2).strip())
            continue

        if current_code:
            entries[current_code].append(line)

    return {
        code: "\n".join(parts).strip()
        for code, parts in entries.items()
        if parts
    }


def lookup_special_provision_code(code: str) -> SpecialProvisionLookup:
    """按特殊规定编号精确查询附录A。"""
    normalized_code = str(code).strip()
    content = load_special_provisions_appendix().get(normalized_code)
    return SpecialProvisionLookup(
        code=normalized_code,
        found=content is not None,
        source_name=str(SPECIAL_PROVISIONS_APPENDIX_PATH),
        content=content,
    )


def lookup_special_provision_codes(codes: List[str]) -> List[SpecialProvisionLookup]:
    """批量精确查询特殊规定编号。"""
    return [lookup_special_provision_code(code) for code in codes]


def metadata_matches_source_filter(
    item: Dict,
    source_filter: Optional[SourceFilter],
) -> bool:
    """判断 metadata/chunk 是否属于指定来源。"""
    if not source_filter or source_filter == "all":
        return True

    source = str(item.get("source", ""))
    section_path = str(item.get("section_path", ""))

    if source_filter == "gb6944":
        return "GB 6944" in source
    if source_filter == "appendix_a":
        return "GB 12268" in source and "附录A" in section_path
    return True


_SOURCE_SEARCH_SPACE_CACHE: Dict[tuple, TextSearchSpace] = {}
_CLAUSE_INDEX_CACHE: Dict[tuple, Dict[tuple, List[Dict]]] = {}


def clear_source_search_space_cache() -> None:
    """清空来源子索引缓存，适合在底层文本索引重建后调用。"""
    _SOURCE_SEARCH_SPACE_CACHE.clear()
    _CLAUSE_INDEX_CACHE.clear()

def _metadata_cache_key(metadata: List[Dict]) -> tuple:
    return (
        id(metadata),
        len(metadata),
    )

def _normalize_section_id(value: str) -> str:
    text = str(value or "").strip()
    match = re.match(r"^(\d+(?:\.\d+)*)", text)
    return match.group(1) if match else text

def _source_key(item: Dict) -> str:
    source = str(item.get("source", ""))
    if "GB 6944" in source:
        return "GB6944"
    if "GB 12268" in source:
        section_path = str(item.get("section_path", ""))
        return "GB12268_APPENDIX_A" if "附录A" in section_path else "GB12268"
    return source

def get_clause_index(metadata: List[Dict]) -> Dict[tuple, List[Dict]]:
    """按 (source_key, section_id) 为 metadata 建立章节索引。"""
    key = _metadata_cache_key(metadata)
    cached = _CLAUSE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    clause_index: Dict[tuple, List[Dict]] = {}
    for meta in metadata:
        section_id = _normalize_section_id(meta.get("section_id", ""))
        if not section_id:
            section_id = _normalize_section_id(meta.get("title", ""))
        if not section_id:
            continue

        index_key = (_source_key(meta), section_id)
        clause_index.setdefault(index_key, []).append(meta)

    _CLAUSE_INDEX_CACHE[key] = clause_index
    return clause_index

def lookup_clause_chunks(
    metadata: List[Dict],
    *,
    source_name: str,
    section_ids: List[str],
) -> List[ClauseLookup]:
    """从 metadata 章节索引中按来源和章节号精确取 chunk。"""
    clause_index = get_clause_index(metadata)
    lookups: List[ClauseLookup] = []
    for section_id in section_ids:
        normalized = _normalize_section_id(section_id)
        chunks = clause_index.get((source_name, normalized), [])
        lookups.append(
            ClauseLookup(
                section_id=normalized,
                found=bool(chunks),
                chunks=[dict(chunk) for chunk in chunks],
            )
        )
    return lookups

def _search_space_cache_key(
    index: faiss.Index,
    metadata: List[Dict],
    source_filter: Optional[SourceFilter],
) -> tuple:
    return (
        id(index),
        id(metadata),
        getattr(index, "ntotal", None),
        len(metadata),
        source_filter,
    )

def get_source_search_space(
    index: faiss.Index,
    metadata: List[Dict],
    source_filter: Optional[SourceFilter],
    *,
    full_bm25_index: Optional[BM25Okapi],
    tokenize: Callable[[str], List[str]],
) -> Optional[TextSearchSpace]:
    """获取指定来源的可复用 FAISS/BM25 子索引。"""
    if not source_filter or source_filter == "all":
        if full_bm25_index is None:
            return None
        return TextSearchSpace(index, metadata, full_bm25_index)

    key = _search_space_cache_key(index, metadata, source_filter)
    cached = _SOURCE_SEARCH_SPACE_CACHE.get(key)
    if cached is not None:
        return cached

    filtered_pairs = [
        (idx, meta)
        for idx, meta in enumerate(metadata)
        if metadata_matches_source_filter(meta, source_filter)
    ]
    if not filtered_pairs:
        return None

    vectors = np.vstack([
        index.reconstruct(idx)
        for idx, _ in filtered_pairs
    ]).astype(np.float32)
    faiss.normalize_L2(vectors)

    subset_index = faiss.IndexFlatIP(vectors.shape[1])
    subset_index.add(vectors)

    subset_metadata = [dict(meta) for _, meta in filtered_pairs]
    subset_bm25 = BM25Okapi([
        tokenize(meta["vector_input"])
        for meta in subset_metadata
    ])

    search_space = TextSearchSpace(subset_index, subset_metadata, subset_bm25)
    _SOURCE_SEARCH_SPACE_CACHE[key] = search_space
    return search_space
