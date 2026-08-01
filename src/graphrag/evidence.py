"""检索证据组装与排序。

本模块负责把 KG、综合映射、文本块和校验证据整理成统一 sources，
并渲染给 LLM 使用的 retrieval_context。
"""

from typing import Dict, List, Optional

from .definitions import Config, QueryRoute
from .comprehensive_query import mapping_to_context, mapping_to_dict
from .index import metadata_matches_source_filter


def chunk_matches_source_filter(chunk: Dict, source_filter: Optional[str]) -> bool:
    """根据 QueryPlan.source_filter 过滤文本证据来源。"""
    return metadata_matches_source_filter(chunk, source_filter)


def apply_source_filter(chunks: List[Dict], source_filter: Optional[str]) -> List[Dict]:
    """只过滤文本证据来源；去重和重新编号交给最终合并阶段。"""
    if not source_filter or source_filter == "all":
        return chunks
    return [
        chunk for chunk in chunks
        if chunk_matches_source_filter(chunk, source_filter)
    ]


def build_retrieval_context(
    query: str,
    route: QueryRoute,
    sources: List[Dict],
) -> str:
    """将检索结果整理成证据上下文，供 run.py 拼接 LLM Prompt。"""
    prompt_parts = [
        "【用户问题】",
        query,
        "",
        "【检索策略】",
        f"- 系统路由: {route}",
    ]

    evidence_parts = []
    validation_sources = [
        source for source in sources
        if source.get("type") == "validation"
    ]
    kg_sources = [source for source in sources if source.get("type") == "kg"]
    mapping_sources = [
        source for source in sources
        if source.get("type") == "comprehensive_mapping"
    ]
    text_sources = [source for source in sources if source.get("type") == "text"]

    has_exact_mapping_evidence = any(
        item.get("source_type") != "text" and item.get("content")
        for source in mapping_sources
        for item in (source.get("mapping") or {}).get("explanation_evidence", [])
    )

    for source in validation_sources:
        evidence_parts.append(
            "【证据A：编号有效性校验】\n"
            f"{str(source.get('content', '')).strip()}"
        )

    for source in kg_sources:
        evidence_parts.append(
            "【证据A：结构化图谱事实】\n"
            f"{str(source.get('content', '')).strip()}"
        )

    for source in mapping_sources:
        evidence_parts.append(
            "【证据C：综合查询字段映射与补充解释】\n"
            f"{str(source.get('content', '')).strip()}"
        )

    should_render_text_evidence = bool(text_sources) or (
        (route in {"direct", "hybrid"} or not kg_sources)
        and not has_exact_mapping_evidence
        and not validation_sources
    )

    if should_render_text_evidence:
        if not text_sources:
            evidence_parts.append(
                "【证据B：非结构化法规条文】\n"
                "未检索到额外的高相关法规文本条款。"
            )
        else:
            text_evidence = ["【证据B：非结构化法规条文】"]
            best_score = text_sources[0].get("similarity_score", 0)
            if best_score < Config.LOW_CONFIDENCE_SCORE:
                text_evidence.append("提示：当前文本召回置信度偏低，回答时需要更谨慎。")

            for index, res in enumerate(text_sources, start=1):
                item = [
                    f"[证据B-{res.get('rank', index)}]",
                    f"来源: {res.get('source', Config.SOURCE_NAME)}",
                    f"章节路径: {res.get('section_path', '')}",
                    f"标题: {res.get('title', '')}",
                    f"正文: {res.get('content', '')}",
                ]
                if res.get("attached_tables"):
                    item.append(f"关联表格数据:\n{res['attached_tables']}")
                text_evidence.append("\n".join(item))

            evidence_parts.append("\n\n".join(text_evidence))

    if not evidence_parts:
        evidence_parts.append("当前没有可用检索证据。")

    prompt_parts.extend([
        "",
        "【检索证据】",
        "\n\n".join(evidence_parts),
    ])

    return "\n".join(prompt_parts)


def build_sources(
    kg_context: Optional[str],
    comprehensive_mapping: Optional[Dict],
    comprehensive_context: Optional[str],
    text_chunks: List[Dict],
    validation_context: Optional[str] = None,
) -> List[Dict]:
    """构造 API sources，同时作为 Prompt 证据渲染的唯一数据源。"""
    sources = []
    if validation_context:
        sources.append({
            "type": "validation",
            "source": "GB 12268-2025 UN编号索引",
            "content": validation_context,
        })
    if kg_context:
        sources.append({
            "type": "kg",
            "source": "GB 12268-2025 Neo4j",
            "content": kg_context,
        })
    if comprehensive_mapping:
        sources.append({
            "type": "comprehensive_mapping",
            "source": "QueryPlan + GB 12268 KG字段映射",
            "content": comprehensive_context or mapping_to_context(comprehensive_mapping),
            "mapping": mapping_to_dict(comprehensive_mapping),
        })
    for item in text_chunks:
        sources.append({
            "type": "text",
            "source": item.get("source", Config.SOURCE_NAME),
            "section_path": item.get("section_path", ""),
            "title": item.get("title", ""),
            "content": item.get("text", ""),
            "rank": item.get("rank"),
            "similarity_score": item.get("similarity_score", 0),
            "attached_tables": item.get("attached_tables", ""),
        })
    return sources


def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for item in chunks:
        key = (
            item.get("source", ""),
            item.get("section_id", ""),
            item.get("section_path", ""),
            item.get("text", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    for index, item in enumerate(unique, start=1):
        item["rank"] = index
    return unique


def merge_ranked_chunks(
    primary_chunks: List[Dict],
    explanation_chunks: List[Dict],
) -> List[Dict]:
    """合并普通文本证据和解释证据，并按证据直接性与相关性重排。"""
    role_priority = {
        "exact_clause": 3,
        "explanation_text": 2,
        "primary_text": 1,
    }

    def sort_key(item: Dict):
        return (
            role_priority.get(item.get("_evidence_role"), 0),
            float(item.get("similarity_score", 0)),
        )

    if not primary_chunks:
        return deduplicate_chunks([
            {**chunk, "_evidence_role": _explanation_chunk_role(chunk)}
            for chunk in explanation_chunks
        ])
    if not explanation_chunks:
        return deduplicate_chunks([
            {**chunk, "_evidence_role": "primary_text"}
            for chunk in primary_chunks
        ])

    tagged_chunks = [
        {**chunk, "_evidence_role": "primary_text"}
        for chunk in primary_chunks
    ]
    tagged_chunks.extend(
        {**chunk, "_evidence_role": _explanation_chunk_role(chunk)}
        for chunk in explanation_chunks
    )
    tagged_chunks.sort(key=sort_key, reverse=True)
    merged = deduplicate_chunks(tagged_chunks)
    for index, item in enumerate(merged, start=1):
        item["rank"] = index
    return merged


def _explanation_chunk_role(chunk: Dict) -> str:
    title = str(chunk.get("title", ""))
    if "精确条文" in title:
        return "exact_clause"
    return "explanation_text"
