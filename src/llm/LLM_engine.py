"""LLM 配置与调用封装。

这个模块集中负责：
- 从环境变量读取 LLM 配置；
- 初始化 OpenAI/兼容 OpenAI 的客户端；
- 调用最终回答模型；
- 调用查询分析模型；
- 调用语义缓存复核模型。

Prompt 文本统一来自 llm.prompt。
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .prompt import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    SEMANTIC_CACHE_VERIFY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_query_analysis_prompt,
    build_semantic_cache_verify_prompt,
)
from graphrag.query_understanding import QueryAnalysis, refine_analysis


LLM_ERROR_PREFIXES = ("LLM 尚未配置", "调用大模型超时", "无法连接到大模型服务", "大模型 API 返回错误",)


def parse_boolean_response(text: str) -> Optional[bool]:
    """解析只应返回 true/false 的轻量 LLM 判断结果。"""
    normalized = (text or "").strip().lower()
    normalized = normalized.strip("` \n\r\t。.!！")
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    match = re.search(r"\b(true|false)\b", normalized)
    if match:
        return match.group(1) == "true"
    return None


@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 90

    @property
    def model_signature(self) -> str:
        return f"{self.model}@temperature={self.temperature}"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            timeout=float(os.getenv("LLM_TIMEOUT", "90")),
        )


@dataclass(frozen=True)
class QueryAnalysisLLMResult:
    analysis: QueryAnalysis
    llm_used: bool
    alias: Optional[str] = None


class LLMEngine:
    """OpenAI/兼容 OpenAI 的模型调用器。"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.client = self._init_client()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def model_signature(self) -> str:
        return self.config.model_signature

    def _init_client(self) -> Optional[OpenAI]:
        api_key = self.config.api_key
        base_url = self.config.base_url

        if not api_key and not base_url:
            return None

        if not api_key:
            api_key = "EMPTY"

        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout)
        return OpenAI(api_key=api_key, timeout=self.config.timeout)

    def generate_answer(self, user_prompt: str) -> str:
        """根据检索证据生成最终答案。"""
        if self.client is None:
            return (
                "LLM 尚未配置，当前仅返回检索证据。"
                "请在 .env 中配置 LLM_API_KEY/OPENAI_API_KEY、LLM_MODEL，"
                "如使用兼容 OpenAI 的本地模型还需要配置 LLM_BASE_URL。"
            )

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
            )
            return completion.choices[0].message.content or ""
        except APITimeoutError:
            return (
                "调用大模型超时。请检查网络或代理配置，或在 .env 中增大 LLM_TIMEOUT，"
                "例如 LLM_TIMEOUT=180。当前检索证据已生成，可以先调用 /prompt 查看检索结果。"
            )
        except APIConnectionError as exc:
            return f"无法连接到大模型服务，请检查 OPENAI_BASE_URL/LLM_BASE_URL、网络或代理配置。错误信息：{exc}"
        except APIError as exc:
            return f"大模型 API 返回错误：{exc}"

    def verify_semantic_candidate(
        self,
        current_question: str,
        cached_question: str,
    ) -> Tuple[bool, bool]:
        """复核 L2 语义缓存模糊候选；返回（通过，是否调用模型）。"""
        if self.client is None:
            return False, False

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": SEMANTIC_CACHE_VERIFY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_semantic_cache_verify_prompt(
                            current_question=current_question,
                            cached_question=cached_question,
                        ),
                    },
                ],
                temperature=0,
                max_tokens=50,
            )
            result = (completion.choices[0].message.content or "").strip().lower()
            parsed = parse_boolean_response(result)
            return parsed is True, True
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            print(f"L2 语义缓存复核失败，按未命中处理: {exc}")
            return False, True
        except Exception as exc:
            print(f"L2 语义缓存复核出现非预期错误，按未命中处理: {exc}")
            return False, True

    def refine_query_analysis(
        self,
        question: str,
        base: QueryAnalysis,
        *,
        enabled: bool,
        threshold: float,
    ) -> QueryAnalysisLLMResult:
        """必要时使用 LLM 将模糊问法规范成专业主体、查询对象和路由。"""
        if not enabled or self.client is None or base.confidence >= threshold:
            return QueryAnalysisLLMResult(analysis=base, llm_used=False)

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": QUERY_ANALYSIS_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_query_analysis_prompt(question, base),
                    },
                ],
                temperature=0,
                max_tokens=220,
            )
            payload = extract_json_object(completion.choices[0].message.content or "")
            if not payload:
                return QueryAnalysisLLMResult(analysis=base, llm_used=True)

            refined = refine_analysis(base, payload)
            alias = str(payload.get("alias") or "").strip() or None
            return QueryAnalysisLLMResult(
                analysis=refined,
                llm_used=True,
                alias=alias,
            )
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            print(f"查询分析 LLM 复核失败，使用规则分析结果: {exc}")
            return QueryAnalysisLLMResult(analysis=base, llm_used=True)
        except Exception as exc:
            print(f"查询分析 LLM 复核出现非预期错误，使用规则分析结果: {exc}")
            return QueryAnalysisLLMResult(analysis=base, llm_used=True)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中尽量稳妥地提取 JSON 对象。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
