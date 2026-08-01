"""答案构造模块。

这里是最终答案的统一出口：
- 能用 KG 模板直接回答的简单事实问题，在这里生成；
- direct / hybrid / 综合查询等需要 LLM 整合的问题，也从这里调用 LLM 生成。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from graphrag.definitions import ROUTES
from graphrag.query_understanding import QueryPlan


@dataclass(frozen=True)
class AnswerBuildResult:
    """答案构造结果。"""

    answer: str
    llm_used: bool
    generation_mode: str


def parse_kg_fields(content: str) -> Dict[str, str]:
    """解析 graph_store 返回的 KG 事实文本。"""
    fields = {}
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ": " not in body:
            continue
        key, value = body.split(": ", 1)
        fields[key.strip()] = value.strip()
    return fields


def build_direct_kg_answer(retrieval: Dict[str, Any], plan: QueryPlan) -> Optional[str]:
    """对简单 KG 事实类问题直接生成模板答案。

    这类问题不需要调用最终回答 LLM，例如：
    - UN1170 的包装类别是什么
    - 汽油的有限数量是多少
    - 某物质属于哪一类危险货物

    注意：这里不再做问题关键词判断；用户“问什么”由
    query_understanding.QueryPlan.analysis.target 决定。
    """
    if retrieval["route"] != "kg":
        return None

    kg_source = next((item for item in retrieval["sources"] if item.get("type") == "kg"), None)
    if not kg_source:
        return None

    content = kg_source.get("content", "")
    if "未检索到" in content:
        return None

    fields = parse_kg_fields(content)
    un_no = fields.get("联合国编号 (UN号)", "")
    name_zh = fields.get("中文正式名称", "")
    name_en = fields.get("英文正式名称", "")
    class_or_division = fields.get("类别或项别", "")
    hazard_class = fields.get("所属危险主类") or fields.get("所属危险主类别", "")
    subsidiary = fields.get("次要危险性", "")
    packing_groups = fields.get("允许使用的所有包装组别", "")
    special_provisions = fields.get("特殊规定条文索引", "")
    limited_quantities = fields.get("有限数量限制", "")
    packing_instruction = fields.get("包装规范指令", "")
    tank_provisions = fields.get("移动罐体特殊规定", "")

    target = plan.analysis.target
    subject = f"{un_no}（{name_zh}）" if un_no and name_zh else (un_no or name_zh or "该危险货物")
    source_line = "信息来源：GB 12268-2025 图数据库节点。"

    if target == "packing_group":
        if not packing_groups:
            return None
        return (
            f"{subject}的包装组别为：{packing_groups}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“允许使用的所有包装组别”为：{packing_groups}。\n"
            f"- 对应包装规范指令：{packing_instruction or '未注明'}。\n\n"
            f"{source_line}"
        )

    if target == "hazard_class":
        if not class_or_division and not hazard_class:
            return None
        answer_value = class_or_division or hazard_class
        return (
            f"{subject}的类别或项别为：{answer_value}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“类别或项别”为：{answer_value}。\n"
            f"- 所属危险主类：{hazard_class or '未注明'}。\n"
            f"- 次要危险性：{subsidiary or '未注明'}。\n\n"
            f"{source_line}"
        )

    if target == "subsidiary_hazard":
        if not subsidiary:
            return None
        return (
            f"{subject}的次要危险性为：{subsidiary}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“次要危险性”为：{subsidiary}。\n\n"
            f"{source_line}"
        )

    if target == "limited_quantities":
        if not limited_quantities:
            return None
        return (
            f"{subject}的有限数量限制为：{limited_quantities}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“有限数量限制”为：{limited_quantities}。\n\n"
            f"{source_line}"
        )

    if target == "packing_instruction":
        if not packing_instruction:
            return None
        return (
            f"{subject}的包装规范指令为：{packing_instruction}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“包装规范指令”为：{packing_instruction}。\n\n"
            f"{source_line}"
        )

    if target == "special_provisions":
        if not special_provisions:
            return None
        return (
            f"{subject}的特殊规定条文索引为：{special_provisions}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“特殊规定条文索引”为：{special_provisions}。\n\n"
            f"{source_line}"
        )

    if target == "portable_tank":
        if not tank_provisions:
            return None
        return (
            f"{subject}的移动罐体特殊规定为：{tank_provisions}。\n\n"
            f"依据：\n- 图数据库中 {un_no or subject} 的“移动罐体特殊规定”为：{tank_provisions}。\n\n"
            f"{source_line}"
        )

    if target == "name":
        return (
            f"{un_no or '该UN编号'}对应的中文正式名称为：{name_zh or '未注明'}；英文正式名称为：{name_en or '未注明'}。\n\n"
            f"依据：\n- 图数据库中该条目的中文、英文正式名称字段如上。\n\n"
            f"{source_line}"
        )

    if hazard_class:
        return (
            f"{subject}的结构化查询结果如下：\n"
            f"- 类别或项别：{class_or_division or hazard_class}\n"
            f"- 所属危险主类：{hazard_class or '未注明'}\n"
            f"- 次要危险性：{subsidiary or '未注明'}\n"
            f"- 包装组别：{packing_groups or '未注明'}\n"
            f"- 有限数量限制：{limited_quantities or '未注明'}\n"
            f"- 包装规范指令：{packing_instruction or '未注明'}\n\n"
            f"{source_line}"
        )

    return None


def build_final_answer(
    question: str,
    retrieval: Dict[str, Any],
    prompt: str,
    llm_engine: Any,
    plan: QueryPlan,
) -> AnswerBuildResult:
    """统一生成最终答案。

    run.py 不再判断“模板答案还是 LLM 答案”，只把问题、检索结果、
    prompt、LLM 引擎和 QueryPlan 交给这里。
    """
    mapping = retrieval.get("comprehensive_mapping")
    if mapping and mapping.get("needs_explanation"):
        answer = llm_engine.generate_answer(prompt)
        return AnswerBuildResult(
            answer=answer,
            llm_used=bool(getattr(llm_engine, "enabled", False)),
            generation_mode="comprehensive_llm",
        )

    direct_answer = build_direct_kg_answer(retrieval, plan)
    if direct_answer:
        return AnswerBuildResult(
            answer=direct_answer,
            llm_used=False,
            generation_mode="kg_template",
        )

    answer = llm_engine.generate_answer(prompt)
    route = retrieval.get("route", "unknown")
    mode = f"{route}_llm" if route in ROUTES else "llm"
    return AnswerBuildResult(
        answer=answer,
        llm_used=bool(getattr(llm_engine, "enabled", False)),
        generation_mode=mode,
    )
