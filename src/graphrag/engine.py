import os
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase

from .settings import apply_runtime_environment

apply_runtime_environment()

from sentence_transformers import SentenceTransformer

from .definitions import Config, QueryRoute
from .comprehensive_query import (
    ExplanationEvidence,
    map_direct_reference,
    map_query_attribute,
    mapping_to_context,
    mapping_to_dict,
)
from .index import (
    clear_source_search_space_cache,
)
from .kg_query import query_kg
from .model_runtime import load_models
from .chunker import (
    attach_table_metadata,
    load_all_tables,
    parse_all_text_files,
)
from .query_understanding import build_query_plan
from . import evidence, explanation_retrieval, text_search


# 全局变量
embedding_model = None
reranker_tokenizer = None
reranker_model = None
faiss_index = None
id_to_metadata = []
bm25_index = None
neo4j_driver = None


def tokenize(text: str) -> List[str]:
    return text_search.tokenize(text)


def build_indices(chunks: List[Dict], model: SentenceTransformer):
    global id_to_metadata, bm25_index
    index, id_to_metadata, bm25_index = text_search.build_indices(chunks, model)
    return index


def save_index_and_metadata(index: faiss.Index, metadata: List[Dict], index_path: str, meta_path: str):
    text_search.save_index_and_metadata(index, metadata, index_path, meta_path)


def load_index_and_metadata(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict]]:
    global bm25_index
    index, metadata, bm25_index = text_search.load_index_and_metadata(index_path, meta_path)
    return index, metadata


def rrf_score(vector_hits: List[int], bm25_hits: List[int], k: int = 60) -> List[Tuple[int, float]]:
    return text_search.rrf_score(vector_hits, bm25_hits, k=k)


def rrf_score_lists(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    return text_search.rrf_score_lists(rankings, k=k)


def extract_query_core(query: str) -> str:
    return text_search.extract_query_core(query)


def lexical_match_score(query: str, meta: Dict) -> float:
    return text_search.lexical_match_score(query, meta)


# ==================== 6. 文本搜索内核 ====================
def expand_query(query: str) -> List[str]:
    return text_search.expand_query(query)


def search_text_chunks(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict],
    top_k: int = 3,
    bm25=None,
) -> List[Dict]:
    active_bm25 = bm25 or bm25_index
    if reranker_tokenizer is None or reranker_model is None:
        return []
    return text_search.search_text_chunks(
        query,
        model,
        index,
        metadata,
        top_k=top_k,
        bm25=active_bm25,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
    )


def chunk_matches_source_filter(chunk: Dict, source_filter: Optional[str]) -> bool:
    return evidence.chunk_matches_source_filter(chunk, source_filter)


def apply_source_filter(chunks: List[Dict], source_filter: Optional[str]) -> List[Dict]:
    return evidence.apply_source_filter(chunks, source_filter)


def search_text_chunks_with_filter(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict],
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    return text_search.search_text_chunks_with_filter(
        query,
        model,
        index,
        metadata,
        top_k=top_k,
        source_filter=source_filter,
        full_bm25_index=bm25_index,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
    )


# ==================== 7. 最终 GraphRAG 混合检索入口 ====================
def build_retrieval_context(
    query: str,
    route: QueryRoute,
    sources: List[Dict],
) -> str:
    return evidence.build_retrieval_context(query, route, sources)


def build_sources(
    kg_context: Optional[str],
    comprehensive_mapping: Optional[Dict],
    comprehensive_context: Optional[str],
    text_chunks: List[Dict],
    validation_context: Optional[str] = None,
) -> List[Dict]:
    return evidence.build_sources(
        kg_context=kg_context,
        comprehensive_mapping=comprehensive_mapping,
        comprehensive_context=comprehensive_context,
        text_chunks=text_chunks,
        validation_context=validation_context,
    )


def _deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    return evidence.deduplicate_chunks(chunks)


def merge_ranked_chunks(
    primary_chunks: List[Dict],
    explanation_chunks: List[Dict],
) -> List[Dict]:
    return evidence.merge_ranked_chunks(primary_chunks, explanation_chunks)


def _explanation_chunk_role(chunk: Dict) -> str:
    title = str(chunk.get("title", ""))
    if "精确条文" in title:
        return "exact_clause"
    return "explanation_text"


def retrieve_explanation_chunks(
    evidence_items: List[ExplanationEvidence],
    top_k_per_query: int = 2,
    model: Optional[SentenceTransformer] = None,
    index: Optional[faiss.Index] = None,
    metadata: Optional[List[Dict]] = None,
) -> List[Dict]:
    """根据综合映射生成的文本查询补充检索法规解释。"""
    active_model = model or embedding_model
    active_index = index or faiss_index
    active_metadata = metadata or id_to_metadata
    if active_model is None or active_index is None or not active_metadata:
        return []

    def search_text(query: str, top_k: int) -> List[Dict]:
        return search_text_chunks(
            query,
            active_model,
            active_index,
            active_metadata,
            top_k,
        )

    return explanation_retrieval.retrieve_explanation_chunks(
        evidence_items,
        metadata=active_metadata,
        search_text=search_text,
        top_k_per_query=top_k_per_query,
    )


def init_engine(rebuild_index: bool = False):
    """初始化检索引擎，供 API 服务启动时调用。"""
    global embedding_model, faiss_index, id_to_metadata, reranker_model, reranker_tokenizer, neo4j_driver

    models_ready = all([embedding_model, reranker_model, reranker_tokenizer])
    index_ready = faiss_index is not None and bool(id_to_metadata) and bm25_index is not None

    if not models_ready:
        models = load_models(Config.EMBEDDING_MODEL, Config.RERANK_MODEL)
        embedding_model = models.embedding
        reranker_tokenizer = models.reranker_tokenizer
        reranker_model = models.reranker

    if neo4j_driver is None:
        print(f"正在建立 Neo4j 连接 ({Config.NEO4J_URI}) ...")
        neo4j_driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))

    if index_ready and not rebuild_index:
        return

    if (
        not rebuild_index
        and os.path.exists(Config.INDEX_SAVE_PATH)
        and os.path.exists(Config.METADATA_SAVE_PATH)
    ):
        print("正在加载已有 FAISS 索引和元数据...")
        faiss_index, id_to_metadata = load_index_and_metadata(Config.INDEX_SAVE_PATH, Config.METADATA_SAVE_PATH)
    else:
        print("正在重建 FAISS 索引和元数据...")
        clear_source_search_space_cache()
        tables = load_all_tables(Config.TABLE_DIR)
        chunks = parse_all_text_files(Config.TEXT_DIR, Config.TEXT_FILE, Config.CHUNK_MAX_LENGTH)
        chunks = attach_table_metadata(chunks, tables)
        faiss_index = build_indices(chunks, embedding_model)
        save_index_and_metadata(faiss_index, id_to_metadata, Config.INDEX_SAVE_PATH, Config.METADATA_SAVE_PATH)


def retrieve_context(query: str, top_k: int = Config.FINAL_TOP_K, query_plan=None) -> Dict:
    """后端 API 使用的非交互式检索入口。"""
    init_engine(rebuild_index=False)

    plan = query_plan or build_query_plan(query)
    route = plan.route
    print(f" [问题路由] 当前问题被判定为: {route}")
    source_filter = getattr(plan, "source_filter", None)

    kg_context = None
    kg_result = None
    comprehensive_mapping = None
    comprehensive_context = None
    validation_context = None
    explanation_chunks: List[Dict] = []
    text_chunks: List[Dict] = []
    text_searched = False

    def run_text_search(reason: str) -> None:
        nonlocal text_chunks, text_searched
        print(reason)
        text_chunks = search_text_chunks_with_filter(
            plan.text_query,
            embedding_model,
            faiss_index,
            id_to_metadata,
            top_k,
            source_filter=source_filter,
        )
        text_searched = True

    if plan.analysis.invalid_un_number:
        validation_context = (
            f"【编号校验提示】未在 GB 12268-2025 品名表有效 UN 编号索引中找到 "
            f"{plan.analysis.invalid_un_number}。该编号被视为无效或超出当前品名表范围，"
            "系统已跳过 Neo4j 查询和文本模糊检索，以避免误召回相近编号或无关条文。"
        )

    if plan.requires_graph:
        kg_result = query_kg(neo4j_driver, plan)
        kg_context = kg_result.context if kg_result else None

    if kg_result and kg_result.fact:
        comprehensive_mapping = map_query_attribute(plan, kg_result.fact)
    if comprehensive_mapping is None:
        comprehensive_mapping = map_direct_reference(plan)

    use_comprehensive_explanation = (
        comprehensive_mapping is not None
        and comprehensive_mapping.has_value
        and comprehensive_mapping.needs_explanation
    )
    graph_subject_missing = (
        plan.requires_graph
        and kg_result is not None
        and not kg_result.found
    )
    allows_text_fallback = plan.requires_text or route == "kg"

    if plan.requires_text and not use_comprehensive_explanation and not graph_subject_missing:
        run_text_search("正在执行文本库：向量 + BM25 + Reranker 混合检索...")

    if use_comprehensive_explanation:
        explanation_chunks = retrieve_explanation_chunks(
            comprehensive_mapping.explanation_evidence,
            top_k_per_query=2,
        )
        explanation_chunks = apply_source_filter(explanation_chunks, source_filter)
        text_chunks = merge_ranked_chunks(text_chunks, explanation_chunks)
        comprehensive_context = mapping_to_context(comprehensive_mapping)

    if (
        plan.requires_graph
        and allows_text_fallback
        and (kg_result is None or not kg_result.found)
        and not text_searched
        and not text_chunks
    ):
        run_text_search("KG 未命中，自动降级为文本检索。")

    sources = build_sources(
        kg_context=kg_context,
        comprehensive_mapping=comprehensive_mapping,
        comprehensive_context=comprehensive_context,
        text_chunks=text_chunks,
        validation_context=validation_context,
    )
    retrieval_context = build_retrieval_context(query, route, sources)

    return {
        "query": query,
        "route": route,
        "analysis": plan.analysis,
        "query_plan": plan,
        "source_filter": source_filter,
        "kg_context": kg_context,
        "kg_fact": kg_result.fact_dict() if kg_result else None,
        "comprehensive_mapping": (
            mapping_to_dict(comprehensive_mapping) if comprehensive_mapping else None
        ),
        "explanation_chunks": explanation_chunks,
        "text_chunks": text_chunks,
        "sources": sources,
        "retrieval_context": retrieval_context,
    }


def close_engine():
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.close()
        neo4j_driver = None


def is_ready() -> bool:
    """返回检索模型、向量索引和元数据是否已经初始化。"""
    return (
        embedding_model is not None
        and reranker_model is not None
        and faiss_index is not None
        and bool(id_to_metadata)
    )


def encode_query(query: str) -> np.ndarray:
    """复用已加载的 BGE-M3，为 L2 语义缓存生成问题向量。"""
    init_engine(rebuild_index=False)
    vector = embedding_model.encode([query], convert_to_numpy=True)
    return np.asarray(vector[0], dtype=np.float32)
