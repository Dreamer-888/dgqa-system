import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graphrag.settings import apply_runtime_environment, settings

apply_runtime_environment()
from graphrag import engine as retrieval
from graphrag.cache import (
    ALIAS_MEMORY_CACHE,
    L1AnswerCache,
    L2SemanticCache,
    L3_ENTITY_CACHE,
    normalize_question,
)
from graphrag.definitions import ROUTES
from graphrag.query_understanding import (
    QueryPlan,
    analyze_query,
    build_query_plan,
)
from llm.LLM_engine import LLMEngine, LLM_ERROR_PREFIXES
from llm.answer import build_final_answer
from llm.prompt import build_user_prompt


REBUILD_INDEX_ON_START = settings.api.rebuild_index


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户自然语言问题")
    top_k: int = Field(default=3, ge=1, le=8, description="文本检索返回条数")
    return_prompt: bool = Field(default=False, description="是否在响应中返回发给 LLM 的 Prompt")


class AskResponse(BaseModel):
    answer: str
    route: str
    sources: List[Dict[str, Any]]
    prompt: Optional[str] = None
    llm_enabled: bool
    llm_used: bool
    from_cache: bool = False
    cache_level: Optional[str] = None
    semantic_similarity: Optional[float] = None
    matched_question: Optional[str] = None
    cache_verified_by_llm: bool = False


class PromptResponse(BaseModel):
    prompt: str
    route: str
    sources: List[Dict[str, Any]]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """统一管理检索引擎和大模型客户端的启动、关闭。"""
    retrieval.init_engine(rebuild_index=REBUILD_INDEX_ON_START)
    try:
        yield
    finally:
        retrieval.close_engine()


app = FastAPI(
    title="Dangerous Goods GraphRAG API",
    description="危险货物知识图谱与 RAG 问答后端",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_engine = LLMEngine()
CACHE_MODEL_SIGNATURE = llm_engine.model_signature
DATA_VERSION = settings.cache.data_version
PROMPT_VERSION = settings.cache.prompt_version

L1_CACHE = L1AnswerCache(
    maxsize=settings.cache.l1_size,
    ttl=settings.cache.l1_ttl,
)
L2_CACHE = L2SemanticCache(
    database_path=settings.cache.l2_db,
    minimum_threshold=settings.cache.l2_min_threshold,
    direct_threshold=settings.cache.l2_direct_threshold,
    max_entries=settings.cache.l2_size,
)
L2_CACHE_ENABLED = settings.cache.l2_enabled
QUERY_ANALYSIS_LLM_ENABLED = settings.query_analysis.llm_enabled
QUERY_ANALYSIS_LLM_THRESHOLD = settings.query_analysis.llm_threshold


def validate_question(question: str) -> str:
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        raise HTTPException(status_code=400, detail="问题不能为空。")
    if len(cleaned) > 500:
        raise HTTPException(status_code=400, detail="问题过长，请控制在 500 字以内。")
    return cleaned


def cache_key(question: str, top_k: int) -> str:
    return "|".join([
        DATA_VERSION,
        PROMPT_VERSION,
        normalize_question(question),
        f"top_k={top_k}",
        f"model={llm_engine.model}",
        f"temperature={llm_engine.temperature}",
    ])


def is_cacheable_response(response: AskResponse) -> bool:
    """错误、超时、未配置 LLM 或无证据结果不进入长期缓存。"""
    return (
        bool(response.answer.strip())
        and bool(response.sources)
        and not response.answer.startswith(LLM_ERROR_PREFIXES)
    )


def save_answer_caches(
    key: str,
    question: str,
    top_k: int,
    response: AskResponse,
    full_prompt: str,
    semantic_question: Optional[str] = None,
    semantic_intent: Optional[str] = None,
    semantic_entity: Optional[str] = None,
) -> None:
    stored = response.model_copy(deep=True)
    stored.prompt = full_prompt
    stored.from_cache = False
    stored.cache_level = None
    stored.semantic_similarity = None
    stored.matched_question = None
    stored.cache_verified_by_llm = False
    payload = stored.model_dump()
    L1_CACHE.put(key, payload)

    if (
        L2_CACHE_ENABLED
        and stored.route in ROUTES
        and is_cacheable_response(stored)
    ):
        L2_CACHE.put(
            question=semantic_question or question,
            route=stored.route,
            response=payload,
            top_k=top_k,
            model=CACHE_MODEL_SIGNATURE,
            data_version=DATA_VERSION,
            prompt_version=PROMPT_VERSION,
            embed=retrieval.encode_query,
            intent=semantic_intent,
            entity=semantic_entity,
        )


def analyze_question_plan(question: str) -> Tuple[QueryPlan, bool]:
    """统一问题处理入口：规则分析，必要时 LLM 规范化，再生成查询计划。"""
    base = analyze_query(question, learned_aliases=ALIAS_MEMORY_CACHE.aliases())
    result = llm_engine.refine_query_analysis(
        question,
        base,
        enabled=QUERY_ANALYSIS_LLM_ENABLED,
        threshold=QUERY_ANALYSIS_LLM_THRESHOLD,
    )
    refined = result.analysis
    if result.alias and refined.entity_name:
        ALIAS_MEMORY_CACHE.put(
            alias=result.alias,
            canonical_name=refined.entity_name,
            source="llm_query_analysis",
            confidence=refined.confidence,
        )
    plan = build_query_plan(question, analysis=refined)
    print(
        f"[问题分析] route={plan.route}, target={plan.analysis.target}, "
        f"subject={plan.analysis.subject}, confidence={plan.analysis.confidence}, "
        f"llm_refined={plan.analysis.refined_by_llm}"
    )
    return plan, result.llm_used


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "retrieval_engine": "ready" if retrieval.is_ready() else "not_initialized",
        "llm_enabled": llm_engine.enabled,
        "llm_model": llm_engine.model if llm_engine.enabled else None,
        "cache": {
            "l1_answer": L1_CACHE.stats(),
            "l2_semantic": L2_CACHE.stats(),
            "l3_entity": L3_ENTITY_CACHE.stats(),
            "alias_memory": ALIAS_MEMORY_CACHE.stats(),
            "data_version": DATA_VERSION,
            "prompt_version": PROMPT_VERSION,
        },
    }


@app.post("/prompt", response_model=PromptResponse)
def build_prompt(request: AskRequest) -> PromptResponse:
    question = validate_question(request.question)
    plan, _ = analyze_question_plan(question)
    context = retrieval.retrieve_context(question, top_k=request.top_k, query_plan=plan)
    prompt = build_user_prompt(context["retrieval_context"], plan=plan)
    return PromptResponse(
        prompt=prompt,
        route=context["route"],
        sources=context["sources"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = validate_question(request.question)
    key = cache_key(question, request.top_k)

    l1_payload = L1_CACHE.get(key)
    if l1_payload is not None:
        cached = AskResponse.model_validate(l1_payload)
        cached.from_cache = True
        cached.cache_level = "L1"
        cached.llm_used = False
        cached.cache_verified_by_llm = False
        cached.prompt = cached.prompt if request.return_prompt else None
        return cached

    plan, analysis_llm_used = analyze_question_plan(question)
    route = plan.route
    semantic_intent = plan.analysis.target
    semantic_entity = plan.analysis.subject or plan.analysis.invalid_un_number or ""
    if L2_CACHE_ENABLED and route in ROUTES:
        candidate = L2_CACHE.find_candidate(
            question=plan.cache_question,
            route=route,
            top_k=request.top_k,
            model=CACHE_MODEL_SIGNATURE,
            data_version=DATA_VERSION,
            prompt_version=PROMPT_VERSION,
            embed=retrieval.encode_query,
            intent=semantic_intent,
            entity=semantic_entity,
        )
        if candidate is not None:
            accepted, verified_by_llm = llm_engine.verify_semantic_candidate(
                current_question=plan.cache_question,
                cached_question=candidate.matched_question,
            )

            if accepted:
                L2_CACHE.accept(candidate)
                cached = AskResponse.model_validate(candidate.response)
                cached.from_cache = True
                cached.cache_level = "L2"
                cached.llm_used = verified_by_llm or analysis_llm_used
                cached.cache_verified_by_llm = verified_by_llm
                cached.semantic_similarity = round(candidate.similarity, 4)
                cached.matched_question = candidate.matched_question
                full_prompt = cached.prompt
                cached.prompt = full_prompt if request.return_prompt else None
                l1_stored = cached.model_copy(deep=True)
                l1_stored.prompt = full_prompt
                L1_CACHE.put(key, l1_stored.model_dump())
                return cached

            L2_CACHE.reject()

    context = retrieval.retrieve_context(question, top_k=request.top_k, query_plan=plan)
    prompt = build_user_prompt(context["retrieval_context"], plan=plan)
    answer_result = build_final_answer(
        question=question,
        retrieval=context,
        prompt=prompt,
        llm_engine=llm_engine,
        plan=plan,
    )
    response = AskResponse(
        answer=answer_result.answer,
        route=context["route"],
        sources=context["sources"],
        prompt=prompt if request.return_prompt else None,
        llm_enabled=llm_engine.enabled,
        llm_used=answer_result.llm_used or analysis_llm_used,
    )
    if is_cacheable_response(response):
        save_answer_caches(
            key, question, request.top_k, response, prompt,
            semantic_question=plan.cache_question,
            semantic_intent=semantic_intent,
            semantic_entity=semantic_entity,
        )
    return response


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="危险货物 GraphRAG 问答后端。")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="启动后端前强制重建 data/text/faiss_index.bin 和 data/text/metadata.json。",
    )
    args = parser.parse_args()
    if args.rebuild:
        os.environ["REBUILD_INDEX"] = "true"
        REBUILD_INDEX_ON_START = True

    reload_enabled = settings.api.reload
    application = "run:app" if reload_enabled else app
    uvicorn.run(
        application,
        host=settings.api.host,
        port=settings.api.port,
        reload=reload_enabled,
    )
