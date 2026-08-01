"""综合查询的解释性证据补充检索。"""

import re
from typing import Callable, Dict, List

from .comprehensive_query import ExplanationEvidence
from .definitions import EXACT_EXPLANATION_CLAUSES, EXPLANATION_SORT_HINTS, Config
from .evidence import deduplicate_chunks
from .index import lookup_clause_chunks


TextSearchFunc = Callable[[str, int], List[Dict]]


def _rule_matches_query(rule: Dict, query: str) -> bool:
    keywords = tuple(rule.get("keywords", ()))
    if rule.get("match") == "all":
        return all(keyword in query for keyword in keywords)
    return any(keyword in query for keyword in keywords)


def metadata_to_exact_chunk(meta: Dict, rank: int = 1, score: float = 999.0) -> Dict:
    return {
        "rank": rank,
        "similarity_score": score,
        "source": meta.get("source", Config.SOURCE_NAME),
        "section_id": meta.get("section_id", ""),
        "section_path": meta.get("section_path", meta.get("title", "")),
        "title": f"[★精确条文] {meta.get('title', '')}",
        "text": meta.get("text", ""),
        "attached_tables": meta.get("attached_tables", ""),
        "raw_refs": meta.get("raw_refs", []),
    }


def exact_explanation_chunks(
    evidence: ExplanationEvidence,
    metadata: List[Dict],
) -> List[Dict]:
    """根据集中规则和 metadata 章节索引精确命中解释条文。"""
    query = evidence.query or ""
    exact: List[Dict] = []

    for rule in EXACT_EXPLANATION_CLAUSES:
        if evidence.source_name != rule["source_name"]:
            continue
        if not _rule_matches_query(rule, query):
            continue

        for result in lookup_clause_chunks(
            metadata,
            source_name=rule["source_name"],
            section_ids=list(rule["section_ids"]),
        ):
            if not result.found:
                continue
            exact.extend(
                metadata_to_exact_chunk(meta, score=float(rule["score"]))
                for meta in result.chunks
            )

    return exact


def source_matches(evidence: ExplanationEvidence, chunk: Dict) -> bool:
    source = chunk.get("source", "")
    section_path = chunk.get("section_path", "")
    if evidence.source_name == "GB6944":
        return "GB 6944" in source
    if evidence.source_name == "GB12268附录A":
        return "GB 12268" in source and "附录A" in section_path
    return True


def explanation_sort_key(evidence: ExplanationEvidence, chunk: Dict) -> float:
    text = " ".join([
        chunk.get("source", ""),
        chunk.get("section_path", ""),
        chunk.get("title", ""),
        chunk.get("text", "")[:300],
    ])
    query = evidence.query or ""
    score = float(chunk.get("similarity_score", 0))

    for hint in EXPLANATION_SORT_HINTS:
        if evidence.source_name != hint["source_name"]:
            continue
        if not any(keyword in query for keyword in hint["keywords"]):
            continue
        for pattern, delta in hint.get("boost_contains", ()):
            if pattern in text:
                score += float(delta)
        for pattern, delta in hint.get("penalty_regex", ()):
            if re.search(pattern, text):
                score += float(delta)

    return score


def retrieve_explanation_chunks(
    evidence_items: List[ExplanationEvidence],
    *,
    metadata: List[Dict],
    search_text: TextSearchFunc,
    top_k_per_query: int = 2,
) -> List[Dict]:
    """根据综合映射生成的文本查询补充检索法规解释。"""
    chunks: List[Dict] = []
    for item in evidence_items:
        if item.source_type != "text" or not item.query:
            continue
        print(f"正在执行综合查询补充检索: {item.query}")
        exact_chunks = exact_explanation_chunks(item, metadata)
        if exact_chunks:
            chunks.extend(exact_chunks[:top_k_per_query])
            continue

        results = search_text(item.query, max(top_k_per_query * 3, top_k_per_query))
        filtered = [chunk for chunk in results if source_matches(item, chunk)]
        filtered.sort(key=lambda chunk: explanation_sort_key(item, chunk), reverse=True)
        chunks.extend(filtered[:top_k_per_query])
    return deduplicate_chunks(chunks)
