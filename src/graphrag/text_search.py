"""文本检索内核。

本模块只负责法规文本的索引构建与向量/BM25/reranker 检索。
它不连接 Neo4j，也不理解最终问答流程。
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import faiss
import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .definitions import (
    COMMON_DG_NAMES,
    DEFINITION_KEYWORDS,
    DEFINITION_QUERY_STOPWORDS,
    DOMAIN_WORDS,
    EXACT_EXPLANATION_CLAUSES,
    LEXICAL_QUERY_STOPWORDS,
    QUERY_CORE_STOPWORDS,
    QUERY_EXPANSION_SYNONYMS,
    TOKEN_STOPWORDS,
    Config,
)
from .index import get_source_search_space
from .query_understanding import extract_un_number


for word in DOMAIN_WORDS:
    jieba.add_word(word)
for name in COMMON_DG_NAMES:
    jieba.add_word(name)


def tokenize(text: str) -> List[str]:
    """中文 BM25 分词：保留英文、数字、UN 编号和危货专业词。"""
    if not text:
        return []
    text = text.lower().replace("－", "-")
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok:
            continue
        if tok in TOKEN_STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def build_indices(chunks: List[Dict], model: SentenceTransformer) -> Tuple[faiss.Index, List[Dict], BM25Okapi]:
    """从文本 chunks 构建 FAISS、metadata 和 BM25 索引。"""
    metadata = []

    texts_for_vector = [chunk["vector_input"] for chunk in chunks]
    print("正在生成文本嵌入向量 (BGE-M3)...")
    embeddings = model.encode(texts_for_vector, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    corpus_tokens = [tokenize(chunk["vector_input"]) for chunk in chunks]
    bm25_index = BM25Okapi(corpus_tokens)

    for chunk in chunks:
        metadata.append({
            "source": chunk.get("source", Config.SOURCE_NAME),
            "section_id": chunk.get("section_id", ""),
            "section_path": chunk.get("section_path", chunk.get("title", "")),
            "title": chunk["title"],
            "text": chunk["text"],
            "vector_input": chunk["vector_input"],
            "level": chunk["level"],
            "is_definition": chunk["is_definition"],
            "attached_tables": chunk.get("metadata", {}).get("attached_tables", ""),
            "raw_refs": chunk.get("metadata", {}).get("raw_refs", []),
        })
    return index, metadata, bm25_index


def save_index_and_metadata(index: faiss.Index, metadata: List[Dict], index_path: str, meta_path: str) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_index_and_metadata(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict], BM25Okapi]:
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    corpus_tokens = [tokenize(meta["vector_input"]) for meta in metadata]
    return index, metadata, BM25Okapi(corpus_tokens)


def rrf_score(vector_hits: List[int], bm25_hits: List[int], k: int = 60) -> List[Tuple[int, float]]:
    return rrf_score_lists([vector_hits, bm25_hits], k=k)


def rrf_score_lists(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    """融合多个有序召回列表，保留每个列表内部排名。"""
    scores: Dict[int, float] = {}
    for hits in rankings:
        seen = set()
        for rank, idx in enumerate(hits):
            if idx == -1 or idx in seen:
                continue
            seen.add(idx)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def extract_query_core(query: str) -> str:
    """抽取检索核心词，避免“是什么/请问”等意图词干扰精确匹配。"""
    core = query.strip()
    for kw in QUERY_CORE_STOPWORDS:
        core = core.replace(kw, " ")
    core = re.sub(r"\s+", "", core)
    return core


def lexical_match_score(query: str, meta: Dict) -> float:
    """给 reranker 增加精确词命中奖励，降低语义相近但关键词不匹配的误排。"""
    core = extract_query_core(query)
    content = " ".join([
        meta.get("section_path", ""),
        meta.get("title", ""),
        meta.get("text", ""),
        meta.get("vector_input", ""),
    ])
    compact_content = re.sub(r"\s+", "", content)
    compact_section_path = re.sub(r"\s+", "", str(meta.get("section_path", "")))
    compact_title = re.sub(r"\s+", "", str(meta.get("title", "")))

    score = 0.0
    if core and (core in content or core in compact_content):
        score += 6.0
        if (
            core in str(meta.get("section_path", ""))
            or core in str(meta.get("title", ""))
            or core in compact_section_path
            or core in compact_title
        ):
            score += 3.0

    query_terms = [
        token for token in tokenize(query)
        if len(token) > 1 and token not in LEXICAL_QUERY_STOPWORDS
    ]
    if query_terms:
        hits = sum(1 for token in query_terms if token in content)
        score += hits / len(query_terms)
        if hits == len(query_terms):
            score += 2.0

    un_number = extract_un_number(query)
    if un_number and un_number in content.upper():
        score += 8.0

    if any(kw in query for kw in DEFINITION_KEYWORDS) and meta.get("is_definition"):
        score += 2.0

    if core and meta.get("is_definition"):
        if f"{core}是指" in compact_content or f"{core}定义为" in compact_content:
            score += 10.0

    return score


def expand_query(query: str) -> List[str]:
    """轻量查询改写：补充同义词，提高召回。"""
    q = query.strip()
    expansions = [q]

    extra = []
    for key, value in QUERY_EXPANSION_SYNONYMS.items():
        if key in q:
            extra.append(value)
    if extra:
        expansions.append(q + " " + " ".join(extra))
    return list(dict.fromkeys(expansions))


def _rule_matches_query(rule: Dict, query: str) -> bool:
    keywords = tuple(rule.get("keywords", ()))
    if rule.get("match") == "all":
        return all(keyword in query for keyword in keywords)
    return any(keyword in query for keyword in keywords)


def search_text_chunks(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict],
    *,
    reranker_tokenizer,
    reranker_model,
    top_k: int = 3,
    bm25: Optional[BM25Okapi] = None,
) -> List[Dict]:
    clean_query = query.strip()
    if not clean_query or bm25 is None:
        return []

    is_definition_query = any(kw in clean_query for kw in DEFINITION_KEYWORDS)
    retrieve_k = max(30, Config.RETRIEVE_TOP_K) if is_definition_query else Config.RETRIEVE_TOP_K
    expanded_queries = expand_query(clean_query)

    vector_rankings: List[List[int]] = []
    for q in expanded_queries:
        q_vec = model.encode([q], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        _, v_indices = index.search(q_vec, retrieve_k)
        vector_rankings.append([idx for idx in v_indices[0] if idx != -1])
    vector_list = [
        idx for idx, _ in rrf_score_lists(vector_rankings)[:retrieve_k]
    ]

    bm25_scores = np.zeros(len(metadata), dtype=np.float32)
    for q in expanded_queries:
        query_tokens = tokenize(q)
        if query_tokens:
            bm25_scores += bm25.get_scores(query_tokens)
    bm25_list = np.argsort(bm25_scores)[::-1][:retrieve_k].tolist()

    merged_hits = rrf_score(vector_list, bm25_list)[:retrieve_k]
    candidate_indices = [idx for idx, _ in merged_hits]

    exact_clause_scores: Dict[int, float] = {}
    if is_definition_query:
        for rule in EXACT_EXPLANATION_CLAUSES:
            if rule["source_name"] != "GB6944":
                continue
            if not _rule_matches_query(rule, clean_query):
                continue
            section_ids = {str(section_id) for section_id in rule["section_ids"]}
            for idx, meta in enumerate(metadata):
                if idx in exact_clause_scores:
                    continue
                if "GB 6944" not in str(meta.get("source", "")):
                    continue
                if str(meta.get("section_id", "")) in section_ids:
                    exact_clause_scores[idx] = float(rule["score"])
                    if idx not in candidate_indices:
                        candidate_indices.insert(0, idx)

        query_terms = set(tokenize(clean_query)) - DEFINITION_QUERY_STOPWORDS
        for idx, meta in enumerate(metadata):
            if meta.get("is_definition", False):
                meta_terms = set(tokenize(meta["vector_input"]))
                if len(query_terms & meta_terms) >= 1 and idx not in candidate_indices:
                    candidate_indices.append(idx)

    candidates = [dict(metadata[idx]) for idx in candidate_indices]
    if not candidates:
        return []

    pairs = [[clean_query, c["vector_input"]] for c in candidates]
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        scores = reranker_model(**inputs).logits.view(-1).float().cpu().numpy()

    for index_offset, score in enumerate(scores):
        candidates[index_offset]["_rerank_score"] = float(score)
        candidates[index_offset]["_lexical_score"] = lexical_match_score(clean_query, candidates[index_offset])
        candidates[index_offset]["_final_score"] = (
            candidates[index_offset]["_rerank_score"]
            + candidates[index_offset]["_lexical_score"]
        )
        candidates[index_offset]["_exact_clause_score"] = exact_clause_scores.get(
            candidate_indices[index_offset],
            0.0,
        )

    if is_definition_query:
        clean_target = extract_query_core(clean_query)

        def get_sort_key(item):
            content = re.sub(
                r"\s+",
                "",
                " ".join([item.get("section_path", ""), item.get("title", ""), item.get("text", "")]),
            )
            target_hit = 1 if clean_target and clean_target in content else 0
            exact_definition = 1 if clean_target and (
                f"{clean_target}是指" in content or f"{clean_target}定义为" in content
            ) else 0
            return (
                item.get("_exact_clause_score", 0.0),
                exact_definition,
                target_hit,
                1 if item.get("is_definition") else 0,
                item["_lexical_score"],
                item["_rerank_score"],
            )

        candidates.sort(key=get_sort_key, reverse=True)
    else:
        candidates.sort(key=lambda item: item["_final_score"], reverse=True)

    results = []
    for rank, meta in enumerate(candidates[:top_k]):
        def_status = "[★定义块]" if meta.get("is_definition") else "[普通块]"
        results.append({
            "rank": rank + 1,
            "similarity_score": meta.get("_final_score", meta["_rerank_score"]),
            "source": meta.get("source", Config.SOURCE_NAME),
            "section_id": meta.get("section_id", ""),
            "section_path": meta.get("section_path", meta.get("title", "")),
            "title": f"{def_status} {meta.get('title', '')}",
            "text": meta["text"],
            "attached_tables": meta.get("attached_tables", ""),
            "raw_refs": meta.get("raw_refs", []),
        })
    return results


def search_text_chunks_with_filter(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict],
    *,
    reranker_tokenizer,
    reranker_model,
    full_bm25_index: Optional[BM25Okapi],
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """在目标来源内完成向量召回、BM25 和 Reranker，避免召回后过滤漏结果。"""
    search_space = get_source_search_space(
        index,
        metadata,
        source_filter,
        full_bm25_index=full_bm25_index,
        tokenize=tokenize,
    )
    if search_space is None:
        return []
    return search_text_chunks(
        query,
        model,
        search_space.faiss_index,
        search_space.metadata,
        top_k=top_k,
        bm25=search_space.bm25_index,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
    )
