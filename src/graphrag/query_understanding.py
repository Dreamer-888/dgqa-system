"""用户问题理解与路由。

这个模块集中处理用户问题的第一步：
- 识别问题主体：UN 编号或危险货物名称；
- 识别查询对象：包装类别、危险类别、特殊规定等；
- 判断查询应走直接查询、KG 事实查询还是综合查询。

后续综合查询模块应优先使用 analyze_query() 的结果，而不是在各处重复写
关键词判断逻辑。
"""

import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional

from .definitions import (
    COMMON_DG_NAMES,
    COMPREHENSIVE_TARGETS,
    DEFINITION_KEYWORDS,
    DIRECT_REFERENCE_KEYWORDS,
    DOMAIN_ENTITIES,
    ENTITY_ALIASES,
    EXPLANATION_KEYWORDS,
    GB12268_REFERENCE_PATTERN,
    GB6944_REFERENCE_PATTERN,
    QUERY_TARGETS,
    ROUTES,
    SUBJECT_TYPES,
    TARGET_CANONICAL_LABELS,
    TARGET_EXPLANATION_HINTS,
    TARGET_KEYWORDS,
    TEXT_ONLY_PREFIXES,
    QueryRoute,
    QueryTarget,
    SourceFilter,
    UN_PATTERN,
)
from .index import load_gb12268_entity_names, load_gb12268_un_numbers


CACHE_REASON_MARKERS = ("为什么", "原因", "依据", "怎么判断", "判定依据")


@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    normalized: str
    route: QueryRoute
    target: QueryTarget
    subject: Optional[str] = None
    subject_type: Optional[Literal["un_number", "entity_name"]] = None
    un_number: Optional[str] = None
    entity_name: Optional[str] = None
    invalid_un_number: Optional[str] = None
    needs_explanation: bool = False
    is_direct_reference: bool = False
    confidence: float = 1.0
    refined_by_llm: bool = False
    refinement_reason: Optional[str] = None
    canonical_question: Optional[str] = None

    @property
    def has_subject(self) -> bool:
        return self.un_number is not None or self.entity_name is not None


@dataclass(frozen=True)
class QueryPlan:
    """供检索流程使用的问题处理结果。

    analysis 保存问题理解的结构化结果；
    route 保持与旧流程兼容；
    graph_query 是用于图谱检索的主体；
    text_query 是用于文本检索的改写查询；
    followup_text_queries 是综合查询时可追加的解释性检索提示。
    """

    analysis: QueryAnalysis
    route: QueryRoute
    graph_query: Optional[str] = None
    text_query: str = ""
    cache_question: str = ""
    requires_graph: bool = False
    requires_text: bool = False
    followup_text_queries: List[str] = field(default_factory=list)
    source_filter: Optional[SourceFilter] = None


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def has_gb6944_reference(query: str) -> bool:
    """识别用户明确把 GB 6944 作为查询对象的表达。"""
    return GB6944_REFERENCE_PATTERN.search(query) is not None


def has_gb12268_reference(query: str) -> bool:
    """识别用户明确提到 GB 12268 的表达。"""
    return GB12268_REFERENCE_PATTERN.search(query) is not None


def extract_un_number(query: str) -> Optional[str]:
    """抽取有效 UN 编号。

    - 用户明确写 UN 前缀时，尊重用户意图，交给 KG 查询判断是否存在；
    - 用户只写裸 4 位数字时，用 GB 12268 品名表中的有效 UN 编号集合
      做校验，避免把标准号、年份、页码等普通数字误识别为 UN 条目。
    """
    valid_un_numbers = load_gb12268_un_numbers()
    for match in UN_PATTERN.finditer(query):
        matched_text = match.group(0)
        explicit_un = re.match(r"(?i)UN\s*[-－]?\s*\d{4}", matched_text.strip()) is not None
        prefix = query[:match.start()].upper()
        compact_prefix = (
            prefix.replace(" ", "")
            .replace("-", "")
            .replace("－", "")
            .replace("/", "")
        )
        if any(compact_prefix.endswith(item) for item in TEXT_ONLY_PREFIXES):
            continue
        candidate = f"UN{match.group(1)}"
        if not explicit_un and valid_un_numbers and candidate not in valid_un_numbers:
            continue
        return candidate
    return None


def detect_invalid_explicit_un_number(query: str) -> Optional[str]:
    """识别用户明确写出的、但不在 GB 12268 品名表中的 UN 编号。"""
    valid_un_numbers = load_gb12268_un_numbers()
    if not valid_un_numbers:
        return None

    for match in UN_PATTERN.finditer(query):
        matched_text = match.group(0)
        explicit_un = re.match(r"(?i)UN\s*[-－]?\s*\d{4}", matched_text.strip()) is not None
        if not explicit_un:
            continue

        candidate = f"UN{match.group(1)}"
        if candidate not in valid_un_numbers:
            return candidate
    return None


def extract_entity_name(
    query: str,
    learned_aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """基于别名和常见品名做最长匹配。

    这一步只做高置信度词典匹配；更开放的名称候选应交给后续 Neo4j
    候选检索/消歧，避免短词误配长词。
    """
    candidates = set(COMMON_DG_NAMES)
    candidates.update(load_gb12268_entity_names())
    candidates.update(ENTITY_ALIASES.keys())
    candidates.update(ENTITY_ALIASES.values())
    if learned_aliases:
        candidates.update(learned_aliases.keys())
        candidates.update(learned_aliases.values())

    matches = [name for name in candidates if name in query]
    if not matches:
        return None
    matched = max(matches, key=len)
    static_alias = ENTITY_ALIASES.get(matched)
    if static_alias:
        return static_alias
    if learned_aliases and matched in learned_aliases:
        return learned_aliases[matched]
    return matched


def resolve_static_entity_alias(query: str) -> Optional[str]:
    """返回用户问题中命中的内置高置信度实体别名。"""
    matches = [alias for alias in ENTITY_ALIASES if alias in query]
    if not matches:
        return None
    return ENTITY_ALIASES[max(matches, key=len)]


def normalize_definition_cache_concept(query: str) -> Optional[str]:
    """把“什么是X / X是什么 / X的定义”归一成同一个 L2 缓存主体。"""
    concept = query.strip()
    concept = GB6944_REFERENCE_PATTERN.sub("", concept)
    concept = GB12268_REFERENCE_PATTERN.sub("", concept)
    for keyword in DEFINITION_KEYWORDS:
        concept = concept.replace(keyword, " ")
    concept = re.sub(r"[的在中关于查询请问请？?。．，,：:；;、\s]+", "", concept)
    return concept or None


def detect_query_target(query: str) -> QueryTarget:
    for target, keywords in TARGET_KEYWORDS:
        if any(keyword in query for keyword in keywords):
            return target
    if re.search(r"第\s*[1-9]类|[1-9](?:\.[1-9])?项", query):
        return "hazard_class"
    return "general"


def estimate_confidence(
    *,
    un_number: Optional[str],
    entity_name: Optional[str],
    target: QueryTarget,
    direct_reference: bool,
) -> float:
    """给规则分析结果一个粗略置信度，用于决定是否需要 LLM 复核。"""
    if direct_reference:
        return 0.95
    if un_number and target != "general":
        return 0.95
    if un_number:
        return 0.88
    if entity_name and target != "general":
        return 0.84
    if entity_name:
        return 0.72
    if target != "general":
        return 0.60
    return 0.50


def has_kg_attribute_intent(target: QueryTarget) -> bool:
    return target in {
        "packing_group",
        "hazard_class",
        "subsidiary_hazard",
        "special_provisions",
        "limited_quantities",
        "excepted_quantities",
        "packing_instruction",
        "special_packing_provisions",
        "portable_tank",
        "name",
    }


def needs_explanation(query: str, target: QueryTarget) -> bool:
    if target in {"definition", "reason", "requirement"}:
        return True
    return any(keyword in query for keyword in EXPLANATION_KEYWORDS)


def is_direct_reference_query(query: str) -> bool:
    """是否为明确询问标准条文/编号/定义的直接查询。"""
    if has_gb6944_reference(query):
        return True
    if has_gb12268_reference(query) and extract_un_number(query) is None and extract_entity_name(query) is None:
        return True
    if any(keyword in query for keyword in DIRECT_REFERENCE_KEYWORDS):
        return extract_un_number(query) is None and extract_entity_name(query) is None
    return False


def analyze_query(
    query: str,
    learned_aliases: Optional[Dict[str, str]] = None,
) -> QueryAnalysis:
    normalized = normalize_query(query)
    gb6944_reference = has_gb6944_reference(normalized)
    gb12268_reference = has_gb12268_reference(normalized)
    un_number = extract_un_number(normalized)
    entity_name = extract_entity_name(normalized, learned_aliases=learned_aliases)
    target = detect_query_target(normalized)
    explanation = needs_explanation(normalized, target)
    invalid_un_number = detect_invalid_explicit_un_number(normalized)
    if invalid_un_number:
        un_number = None
        entity_name = None

    if gb6944_reference:
        direct_reference = True
        un_number = None
        entity_name = None
    elif gb12268_reference and un_number is None and entity_name is None:
        direct_reference = True
    elif any(keyword in normalized for keyword in DIRECT_REFERENCE_KEYWORDS):
        direct_reference = un_number is None and entity_name is None
    else:
        direct_reference = False
    confidence = estimate_confidence(
        un_number=un_number,
        entity_name=entity_name,
        target=target,
        direct_reference=direct_reference,
    )

    subject = un_number or entity_name
    subject_type: Optional[Literal["un_number", "entity_name"]]
    if un_number:
        subject_type = "un_number"
    elif entity_name:
        subject_type = "entity_name"
    else:
        subject_type = None

    if not subject or direct_reference:
        route: QueryRoute = "direct"
    elif explanation or target in COMPREHENSIVE_TARGETS:
        route = "hybrid"
    elif has_kg_attribute_intent(target):
        route = "kg"
    else:
        route = "hybrid"

    analysis = QueryAnalysis(
        original=query,
        normalized=normalized,
        route=route,
        target=target,
        subject=subject,
        subject_type=subject_type,
        un_number=un_number,
        entity_name=entity_name,
        invalid_un_number=invalid_un_number,
        needs_explanation=explanation,
        is_direct_reference=direct_reference,
        confidence=confidence,
    )
    return replace(analysis, canonical_question=build_cache_question(analysis))


def build_cache_question(analysis: QueryAnalysis) -> str:
    """生成更适合 L2 语义缓存的规范化问题。

    原问题仍用于最终回答；L2 的向量问题尽量使用“专业主体 + 标准查询对象”
    的自然语言表达，降低俗称、语序差异导致的缓存碎片。
    """
    if analysis.target == "special_provisions" and not analysis.subject:
        codes = re.findall(r"(?<!\d)(\d{1,4})(?!\d)", analysis.normalized)
        if codes:
            return f"特殊规定{'/'.join(codes)}的含义"

    label = TARGET_CANONICAL_LABELS.get(analysis.target, "相关信息")
    has_reason_marker = any(keyword in analysis.normalized for keyword in CACHE_REASON_MARKERS)
    if analysis.subject:
        if analysis.target == "reason":
            if re.search(r"第\s*[1-9]类|[1-9](?:\.[1-9])?项|危险类别|危险货物类别|项别|分类", analysis.normalized):
                return f"{analysis.subject}的危险类别判定依据"
            if any(keyword in analysis.normalized for keyword in ("包装组", "包装类别", "包装等级")):
                return f"{analysis.subject}的包装类别判定依据"
            return f"{analysis.subject}的判定依据"
        if has_reason_marker and analysis.target not in {"definition", "requirement"}:
            return f"{analysis.subject}的{label}判定依据"
        return f"{analysis.subject}的{label}"
    if analysis.target == "definition":
        concept = normalize_definition_cache_concept(analysis.normalized)
        if concept:
            return f"{concept}的定义"
    if analysis.target != "general":
        return f"{analysis.normalized}，查询对象：{label}"
    return analysis.normalized


def refine_analysis(base: QueryAnalysis, llm_result: Dict[str, object]) -> QueryAnalysis:
    """用 LLM 输出的结构化结果修正规则分析。

    只接受白名单字段和值；缺失或非法字段会保留规则分析结果，避免 LLM
    输出格式漂移影响主流程。
    """
    route = str(llm_result.get("route") or base.route)
    target = str(llm_result.get("target") or base.target)
    subject_type = llm_result.get("subject_type") or base.subject_type
    subject = llm_result.get("subject") or base.subject
    un_number = llm_result.get("un_number") or base.un_number
    entity_name = llm_result.get("entity_name") or base.entity_name
    reason = llm_result.get("reason")

    if route not in ROUTES:
        route = base.route
    if target not in QUERY_TARGETS:
        target = base.target
    if subject_type not in SUBJECT_TYPES:
        subject_type = base.subject_type

    subject = str(subject).strip() if subject else None
    un_number = str(un_number).strip().upper() if un_number else None
    entity_name = str(entity_name).strip() if entity_name else None

    if un_number and not un_number.startswith("UN"):
        digits = "".join(ch for ch in un_number if ch.isdigit())
        un_number = f"UN{digits}" if len(digits) == 4 else base.un_number

    if subject_type == "un_number":
        subject = un_number or subject
        entity_name = None
    elif subject_type == "entity_name":
        subject = entity_name or subject
        un_number = None

    static_entity_alias = resolve_static_entity_alias(base.normalized)
    if static_entity_alias:
        subject = static_entity_alias
        subject_type = "entity_name"
        entity_name = static_entity_alias
        un_number = None
    elif (
        base.entity_name
        and base.entity_name in base.normalized
        and entity_name != base.entity_name
    ):
        subject = base.entity_name
        subject_type = "entity_name"
        entity_name = base.entity_name
        un_number = None
    elif is_gb6944_concept_query(base):
        subject = None
        subject_type = None
        entity_name = None
        un_number = None

    if base.invalid_un_number:
        route = "direct"
        subject = None
        subject_type = None
        un_number = None
        entity_name = None
    elif base.is_direct_reference:
        route = "direct"
    elif not subject:
        route = "direct"
    elif needs_explanation(base.normalized, target) or target in COMPREHENSIVE_TARGETS:
        route = "hybrid"
    elif has_kg_attribute_intent(target):
        route = "kg"
    else:
        route = "hybrid"

    refined = replace(
        base,
        route=route,  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        subject=subject,
        subject_type=subject_type,  # type: ignore[arg-type]
        un_number=un_number,
        entity_name=entity_name,
        needs_explanation=needs_explanation(base.normalized, target),  # type: ignore[arg-type]
        confidence=max(base.confidence, 0.88),
        refined_by_llm=True,
        refinement_reason=str(reason).strip() if reason else None,
    )
    return replace(refined, canonical_question=build_cache_question(refined))


def build_direct_text_query(analysis: QueryAnalysis) -> str:
    """直接查询的文本检索语句。

    直接查询通常已经包含标准号、章节号、表号或特殊规定号，不做激进改写，
    防止把用户明确指定的条文编号冲淡。
    """
    return analysis.normalized


def build_followup_text_queries(analysis: QueryAnalysis) -> List[str]:
    """为综合查询准备解释性文本检索提示。

    当前阶段先根据查询对象生成稳定提示；等图谱返回具体字段值后，
    综合查询模块可在这些提示中追加“特殊规定144”“例外数量E1”等精确值。
    """
    hint = TARGET_EXPLANATION_HINTS.get(analysis.target, "")
    if not hint:
        return []

    if analysis.subject:
        return [f"{analysis.subject} {hint}", hint]
    return [hint]


def build_text_query(analysis: QueryAnalysis) -> str:
    """生成当前检索流程使用的主文本查询。"""
    if analysis.route == "direct":
        return build_direct_text_query(analysis)

    followups = build_followup_text_queries(analysis)
    if analysis.route == "hybrid" and followups:
        return f"{analysis.normalized} {followups[0]}"

    return analysis.normalized


def is_gb6944_concept_query(analysis: QueryAnalysis) -> bool:
    """识别没有具体主体的危险货物分类/概念定义问题。"""
    query = analysis.normalized
    if analysis.subject:
        return False
    if analysis.target in {"packing_group", "hazard_class"}:
        return True
    if analysis.target != "definition":
        return False
    if any(entity in query for entity in DOMAIN_ENTITIES):
        return True
    return re.search(r"第\s*[1-9]类|[1-9](?:\.[1-9])?项", query) is not None


def build_source_filter(analysis: QueryAnalysis) -> Optional[SourceFilter]:
    """生成文本证据来源过滤器。

    gb6944：只允许 GB 6944 文本；
    appendix_a：只允许 GB 12268 附录A；
    all：GB 6944 和 GB 12268 附录A 都允许；
    None：不限制来源。
    """
    if has_gb6944_reference(analysis.normalized):
        return "gb6944"
    if analysis.target == "special_provisions" or "附录A" in analysis.normalized:
        return "appendix_a"
    if is_gb6944_concept_query(analysis):
        return "gb6944"
    if analysis.route == "hybrid":
        return "all"
    return None


def build_query_plan(query: str, analysis: Optional[QueryAnalysis] = None) -> QueryPlan:
    """生成统一查询计划，作为后续直接查询/综合查询的入口。"""
    analysis = analysis or analyze_query(query)
    invalid_un_number = analysis.invalid_un_number is not None
    route: QueryRoute = "direct" if invalid_un_number else analysis.route
    return QueryPlan(
        analysis=analysis,
        route=route,
        graph_query=None if invalid_un_number else analysis.subject,
        text_query=build_text_query(analysis),
        cache_question=analysis.canonical_question or build_cache_question(analysis),
        requires_graph=False if invalid_un_number else route in {"kg", "hybrid"},
        requires_text=False if invalid_un_number else route in {"direct", "hybrid"},
        followup_text_queries=build_followup_text_queries(analysis),
        source_filter=build_source_filter(analysis),
    )
