"""Prompt 模板集中管理。

这个模块只负责生成提示词文本，不直接调用 LLM，也不读写缓存。
"""

import re
from typing import Dict, List, Optional

from graphrag.definitions import (
    EXPLICIT_KG_TARGET_FIELDS,
    KG_BASE_FIELDS,
    KG_EVIDENCE_TITLE,
    KG_KEY_FIELDS,
    QueryTarget,
)
from graphrag.query_understanding import QueryAnalysis, QueryPlan


SYSTEM_PROMPT = """你是危险货物法规知识问答系统的专业助手。
你的回答必须严格依据后端提供的【检索证据】，不要编造未出现在证据中的法规编号、UN编号、类别、包装组、数量限制或操作要求。
如果结构化图谱事实与文本法规条文同时存在，事实性字段优先采用图谱事实数据，解释性内容可以用文本条文补充。
如果证据不足以回答，请明确说明“当前检索证据不足”，并指出还需要查询的资料类型。

回答格式要求：
1. 先给出直接结论。
2. 再用要点列出依据。
3. 最后标注信息来源，例如“GB 12268-2025 图数据库节点”或“GB 6944-2025 条文路径”。
4. 不要输出检索排名、重排分数、系统调试信息。
5. 如果用户询问“为什么、原因、依据、怎么判定”，必须说明判定依据或原因，不能只回答“是的/属于/不属于”。
6. 只使用与用户查询对象直接相关的字段作为依据；例如询问类别判定依据时，不要把特殊规定、例外数量、包装规范等无关字段当作原因。
"""


QUERY_ANALYSIS_SYSTEM_PROMPT = "你是严格的查询分析器，只输出一个 JSON 对象。"


SEMANTIC_CACHE_VERIFY_SYSTEM_PROMPT = (
    "你是严格的语义缓存安全复核器，只输出 true 或 false。"
)


def parse_kg_evidence_fields(kg_text: str) -> Dict[str, str]:
    """解析结构化图谱证据中的字段行。

    graph_store 当前输出格式以 ``- 字段: 值`` 为主。这里采用宽松解析，
    只提取这种稳定字段，忽略标题、空行和其他说明。
    """
    fields: Dict[str, str] = {}
    for line in kg_text.splitlines():
        line = line.strip()
        if not line.startswith("- ") or ": " not in line:
            continue
        key, value = line[2:].split(": ", 1)
        fields[key.strip()] = value.strip()
    return fields


def kg_fields_for_target(target: Optional[QueryTarget]) -> List[str]:
    """根据查询对象选择需要交给 LLM 的 KG 字段。

    只有当 target 明确对应 Neo4j 中的具体属性时才裁剪字段；
    reason/definition/requirement/general 这类开放问题需要更多上下文，
    因此回退为完整 KG 关键字段，避免把有用依据提前裁掉。
    """
    if not target or target not in EXPLICIT_KG_TARGET_FIELDS:
        return KG_KEY_FIELDS

    selected: List[str] = []
    for key in KG_BASE_FIELDS + EXPLICIT_KG_TARGET_FIELDS[target]:
        if key not in selected:
            selected.append(key)
    return selected


def compact_kg_evidence(
    kg_text: str,
    *,
    target: Optional[QueryTarget] = None,
    max_value_chars: int = 120,
) -> str:
    """把 KG 查询结果压缩为适合放入 Prompt 的短证据。

    KG 节点字段很多，但单个问题通常只需要少数事实字段。这个函数不改变
    原始 sources，只压缩最终发给 LLM 的 evidence 文本，从而降低 token 消耗。
    """
    if not kg_text.strip():
        return kg_text

    if "未检索到" in kg_text:
        return kg_text.strip()

    fields = parse_kg_evidence_fields(kg_text)
    if not fields:
        return truncate_text(kg_text.strip(), max_chars=800)

    allowed_fields = kg_fields_for_target(target)
    lines: List[str] = []
    for key in allowed_fields:
        value = fields.get(key)
        if not value:
            continue
        lines.append(f"- {key}: {truncate_text(value, max_chars=max_value_chars)}")

    omitted = len([key for key in fields if key not in allowed_fields])
    if omitted > 0:
        lines.append(f"- 已省略与当前查询对象无直接关系的图谱字段: {omitted}项")

    return "\n".join(lines) if lines else truncate_text(kg_text.strip(), max_chars=800)


def truncate_text(text: str, *, max_chars: int) -> str:
    """按字符数截断文本，保留语义上比较清楚的省略标记。"""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "……"


def compact_kg_evidence_in_context(
    retrieval_context: str,
    *,
    plan: Optional[QueryPlan] = None,
) -> str:
    """压缩检索上下文中的 KG 证据块。

    当前 retrieval_context 是已经拼好的字符串，因此这里做字符串级处理：
    找到 ``【证据A：结构化图谱事实】`` 到下一个 ``【证据X`` 之间的内容，
    只替换这一小段，避免影响文本检索证据和综合映射证据。
    """
    if KG_EVIDENCE_TITLE not in retrieval_context:
        return retrieval_context

    pattern = re.compile(
        rf"({re.escape(KG_EVIDENCE_TITLE)}\n)(.*?)(?=\n\n【证据[BCDEFG]：|\Z)",
        re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        title = match.group(1)
        kg_body = match.group(2)
        target = plan.analysis.target if plan else None
        return title + compact_kg_evidence(kg_body, target=target)

    return pattern.sub(replace, retrieval_context, count=1)


def build_user_prompt(retrieval_context: str, plan: Optional[QueryPlan] = None) -> str:
    """最终回答阶段发送给 LLM 的用户提示词。"""
    retrieval_context = compact_kg_evidence_in_context(retrieval_context, plan=plan)
    return (
        f"{retrieval_context}\n\n"
        "【请生成最终答案】\n"
        "请根据以上检索证据回答用户问题。"
    )


def build_query_analysis_prompt(question: str, base: QueryAnalysis) -> str:
    """模糊查询分析阶段的结构化抽取提示词。"""
    return (
        "你是危险货物法规问答系统的查询分析器。请根据用户问题，"
        "把模糊问法规范成结构化 JSON。只能输出 JSON，不要解释。\n\n"
        "可选 route：direct, kg, hybrid。\n"
        "可选 target：packing_group, hazard_class, subsidiary_hazard, "
        "special_provisions, limited_quantities, excepted_quantities, "
        "packing_instruction, special_packing_provisions, portable_tank, "
        "name, definition, reason, requirement, general。\n"
        "subject_type 只能是 un_number、entity_name 或 null。\n\n"
        "判断原则：\n"
        "1. GB6944、GB12268、章节号、附录、表格、特殊规定编号本身的解释，通常 route=direct。\n"
        "2. UN编号或具体品名 + 结构化字段查询，通常 route=kg。\n"
        "3. 具体品名/UN + 原因、依据、解释、要求，通常 route=hybrid。\n"
        "4. 如果用户使用俗称，请尽量改为 GB 12268 可能采用的专业名称；"
        "例如酒精可规范为乙醇，酒精溶液可规范为乙醇溶液。\n"
        "5. 不确定时不要强行编造 UN 编号。\n\n"
        "输出字段：route, target, subject, subject_type, un_number, entity_name, alias, reason。\n"
        "alias 表示用户问题中的俗称或非标准称呼；如果没有则为 null。\n\n"
        f"用户问题：{question}\n"
        f"规则初判：route={base.route}, target={base.target}, "
        f"subject={base.subject}, subject_type={base.subject_type}, confidence={base.confidence}\n"
    )


def build_semantic_cache_verify_prompt(
    current_question: str,
    cached_question: str,
) -> str:
    """L2 语义缓存模糊命中复核提示词。"""
    return (
        "判断下面两个危险货物法规问题能否安全复用完全相同的答案。"
        "只有实体、询问属性和意图均一致时才返回 true；"
        "任何不确定、范围差异或属性差异都返回 false。只输出 true 或 false。\n"
        f"问题A：{cached_question}\n"
        f"问题B：{current_question}"
    )
