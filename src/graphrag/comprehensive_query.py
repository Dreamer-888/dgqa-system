"""综合查询字段映射。

这个模块负责把 QueryPlan 中的查询对象映射到：
- Neo4j 返回的结构化字段；
- 字段值；
- 需要进一步解释的资料来源；
- 后续文本检索查询词或表格查询结果。

它不直接调用 Neo4j，也不直接生成最终回答。
"""

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .definitions import (
    EXCEPTED_QUANTITY_TABLE_PATH,
    TARGET_FIELD_MAP,
    QueryTarget,
)
from .graph_store import DangerousGoodFact
from .index import lookup_special_provision_codes as lookup_special_provision_index
from .query_understanding import QueryPlan


@dataclass(frozen=True)
class ExplanationEvidence:
    """字段值的进一步解释来源。"""

    source_type: str
    source_name: str
    query: Optional[str] = None
    content: Optional[str] = None


@dataclass(frozen=True)
class AttributeMappingResult:
    """QueryPlan.target 到 KG 字段和解释来源的映射结果。"""

    target: QueryTarget
    label: str
    kg_field: Optional[str]
    values: List[str]
    needs_explanation: bool
    explanation_evidence: List[ExplanationEvidence] = field(default_factory=list)
    note: Optional[str] = None

    @property
    def has_value(self) -> bool:
        return bool(self.values)


def resolve_effective_target(plan: QueryPlan) -> QueryTarget:
    """把“为什么/依据”类问题落到实际被解释的 KG 字段上。

    例如“为什么 UN1203 属于第3类”表面 target 是 reason，
    但实际要解释的是类别或项别，因此应使用 hazard_class 的字段映射和
    GB6944 解释来源。
    """
    target = plan.analysis.target
    if target != "reason":
        return target

    query = plan.analysis.normalized
    if any(keyword in query for keyword in ["属于", "第3类", "哪一类", "哪类", "类别", "项别", "分类"]):
        return "hazard_class"
    if any(keyword in query for keyword in ["包装组", "包装类别", "包装等级"]):
        return "packing_group"
    if "次要危险" in query:
        return "subsidiary_hazard"
    if "特殊规定" in query:
        return "special_provisions"
    if "例外数量" in query:
        return "excepted_quantities"
    return target


def _fact_value(fact: DangerousGoodFact | Dict[str, Any], field_name: str) -> Any:
    if isinstance(fact, dict):
        return fact.get(field_name)
    return getattr(fact, field_name, None)


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return not any(not _is_empty_value(item) for item in value)
    text = str(value).strip()
    return not text or text in {"-", "—", "无", "nan", "NaN", "None"}


def split_field_values(value: Any) -> List[str]:
    """拆分 KG 字段值，保留罗马数字、UN条目代码、E编码等。"""
    if _is_empty_value(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if not _is_empty_value(item)]

    text = str(value).strip()
    if not text:
        return []

    parts = re.split(r"\s*/\s*|[，,、；;]\s*|\s+(?=[A-Z]?\d{1,4}\b)|\s+(?=E[0-5]\b)", text)
    values = [part.strip() for part in parts if not _is_empty_value(part)]
    return values or [text]


def extract_special_provision_codes(value: Any) -> List[str]:
    """从特殊规定字段提取编号。"""
    text = " ".join(split_field_values(value))
    return re.findall(r"(?<!\d)(\d{1,4})(?!\d)", text)


def extract_explicit_special_provision_codes(query: str) -> List[str]:
    """从用户问题中抽取“特殊规定”后明确跟随的编号，避免误取 UN 编号。"""
    codes: List[str] = []
    pattern = re.compile(r"特殊规定\s*((?:\d{1,4})(?:\s*(?:/|、|,|，|和|及)\s*\d{1,4})*)")
    for match in pattern.finditer(query):
        codes.extend(re.findall(r"(?<!\d)(\d{1,4})(?!\d)", match.group(1)))
    return unique_preserve_order(codes)


def unique_preserve_order(values: List[str]) -> List[str]:
    """去重并保留原顺序。"""
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_excepted_quantity_codes(value: Any) -> List[str]:
    """从例外数量字段提取 E0-E5 编码。"""
    text = " ".join(split_field_values(value)).upper()
    return re.findall(r"\bE[0-5]\b", text)


@lru_cache(maxsize=4)
def load_excepted_quantity_table(path: Path = EXCEPTED_QUANTITY_TABLE_PATH) -> Dict[str, Dict[str, str]]:
    """加载例外数量编码表。"""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return {
            row.get("编码", "").strip().upper(): {
                key: (value or "").strip()
                for key, value in row.items()
                if key
            }
            for row in reader
            if row.get("编码")
        }


def lookup_excepted_quantity_codes(codes: List[str]) -> List[ExplanationEvidence]:
    """根据 E0-E5 编码查询例外数量表。"""
    table = load_excepted_quantity_table()
    evidence = []
    for code in codes:
        row = table.get(code.upper())
        if not row:
            evidence.append(
                ExplanationEvidence(
                    source_type="table",
                    source_name=str(EXCEPTED_QUANTITY_TABLE_PATH),
                    query=code,
                    content=f"未在例外数量编码表中找到 {code}。",
                )
            )
            continue

        inner = row.get("每件内包装的最大净装载量（固体单位为 g，液体和气体单位为 mL）", "")
        outer = row.get("每件外包装的最大净装载量（固体单位为 g，液体和气体单位为 mL；在混装的情况下为 g 和 mL 的总和）", "")
        evidence.append(
            ExplanationEvidence(
                source_type="table",
                source_name=str(EXCEPTED_QUANTITY_TABLE_PATH),
                query=code,
                content=f"{code}：每件内包装最大净装载量为 {inner}；每件外包装最大净装载量为 {outer}。",
            )
        )
    return evidence


def lookup_special_provision_codes(codes: List[str]) -> List[ExplanationEvidence]:
    """按特殊规定编号精确查询附录A。

    特殊规定编号是附录A的主键。只要问题或KG字段给出了明确编号，
    就必须走精确索引；编号不存在时返回“未找到”的精确结论，
    不再退化为向量相似检索，避免把 999 误召回为其他相近条目。
    """
    evidence: List[ExplanationEvidence] = []

    for result in lookup_special_provision_index(codes):
        if result.found and result.content:
            evidence.append(
                ExplanationEvidence(
                    source_type="appendix_exact",
                    source_name=result.source_name,
                    query=result.code,
                    content=f"特殊规定{result.code}：{result.content}",
                )
            )
        else:
            evidence.append(
                ExplanationEvidence(
                    source_type="appendix_exact_missing",
                    source_name=result.source_name,
                    query=result.code,
                    content=f"未在GB 12268附录A精确索引中找到特殊规定{result.code}。",
                )
            )

    return evidence


def build_text_evidence_queries(
    plan: QueryPlan,
    target: QueryTarget,
    values: List[str],
) -> List[ExplanationEvidence]:
    """构造需要交给文本检索的解释性查询。"""
    subject = plan.analysis.subject or plan.graph_query or ""

    if target == "hazard_class":
        queries = [f"GB 6944 {value} 危险货物类别 定义 项别" for value in values]
        queries.append("GB 6944 危险货物类别 项别 分类")
        source_name = "GB6944"
    elif target == "subsidiary_hazard":
        queries = [f"GB 6944 次要危险性 {value} 危险性先后顺序" for value in values]
        queries.append("GB 6944 次要危险性 危险性先后顺序")
        source_name = "GB6944"
    elif target == "packing_group":
        value_text = " ".join(values)
        queries = [
            f"GB 6944 包装类别 包装组 {value_text} 划分依据",
            "GB 6944 包装类别 包装组 危险性程度",
        ]
        source_name = "GB6944"
    elif target == "special_provisions":
        codes = extract_special_provision_codes(values)
        # 特殊规定编号是附录A的唯一标识，只走精确索引；
        # 编号不存在时返回明确的 missing 证据，不再走向量兜底。
        if codes:
            return lookup_special_provision_codes(codes)
        queries = [f"GB 12268 附录A 特殊规定 {subject}".strip()]
        source_name = "GB12268附录A"
    else:
        return []

    return [
        ExplanationEvidence(
            source_type="text",
            source_name=source_name,
            query=query,
        )
        for query in queries
        if query.strip()
    ]


def map_query_attribute(
    plan: QueryPlan,
    kg_fact: DangerousGoodFact | Dict[str, Any] | None,
) -> AttributeMappingResult:
    """根据 QueryPlan 从 KG fact 中提取目标字段，并映射解释来源。"""
    target = plan.analysis.target
    effective_target = resolve_effective_target(plan)
    config = TARGET_FIELD_MAP.get(effective_target)
    if not config:
        return AttributeMappingResult(
            target=target,
            label=str(target),
            kg_field=None,
            values=[],
            needs_explanation=False,
            note="当前查询对象暂未配置综合查询字段映射。",
        )

    label = config["label"]
    field_name = config["field"]
    if kg_fact is None:
        return AttributeMappingResult(
            target=target,
            label=label,
            kg_field=field_name,
            values=[],
            needs_explanation=True,
            note="KG 查询未返回结构化事实，无法提取目标属性值。",
        )

    raw_value = _fact_value(kg_fact, field_name)
    values = split_field_values(raw_value)
    evidence: List[ExplanationEvidence] = []

    note = None
    if effective_target == "special_provisions":
        kg_codes = unique_preserve_order(extract_special_provision_codes(values))
        explicit_codes = extract_explicit_special_provision_codes(plan.analysis.normalized)
        lookup_codes = explicit_codes or kg_codes
        if lookup_codes:
            evidence.extend(lookup_special_provision_codes(lookup_codes))
        missing_from_kg = [code for code in explicit_codes if code not in kg_codes]
        if missing_from_kg and kg_codes:
            note = (
                f"用户问题显式提到特殊规定 {'/'.join(missing_from_kg)}，"
                f"但 KG 中 {plan.analysis.subject or plan.graph_query or '该条目'} "
                f"适用的特殊规定为 {'/'.join(kg_codes)}；回答时应区分条目适用编号与用户指定编号。"
            )
    elif config["source"] == "GB12268_EXCEPTED_QUANTITY_TABLE":
        codes = extract_excepted_quantity_codes(values)
        evidence.extend(lookup_excepted_quantity_codes(codes))
    else:
        evidence.extend(build_text_evidence_queries(plan, effective_target, values))

    return AttributeMappingResult(
        target=target,
        label=label,
        kg_field=field_name,
        values=values,
        needs_explanation=bool(evidence),
        explanation_evidence=evidence,
        note=note,
    )


def map_direct_reference(plan: QueryPlan) -> Optional[AttributeMappingResult]:
    """处理不依赖 KG 主体的直接编号类查询。

    目前主要用于“特殊规定243是什么意思”这种问题：特殊规定编号本身就是
    附录A主键，应直接精确查询附录A，而不是走向量相似度。
    """
    if plan.analysis.target != "special_provisions":
        return None

    codes = extract_explicit_special_provision_codes(plan.analysis.normalized)
    if not codes:
        return None

    evidence = lookup_special_provision_codes(codes)
    return AttributeMappingResult(
        target=plan.analysis.target,
        label="特殊规定",
        kg_field=None,
        values=codes,
        needs_explanation=bool(evidence),
        explanation_evidence=evidence,
    )


def mapping_to_context(mapping: AttributeMappingResult) -> str:
    """把映射结果格式化成便于调试/后续 prompt 使用的文本。"""
    value_text = " / ".join(mapping.values) if mapping.values else "未提取到"
    lines = [
        "【综合查询字段映射】",
        f"- 查询对象: {mapping.label} ({mapping.target})",
        f"- KG字段: {mapping.kg_field or '无'}",
        f"- KG字段值: {value_text}",
    ]
    if mapping.note:
        lines.append(f"- 备注: {mapping.note}")
    if mapping.explanation_evidence:
        lines.append("- 进一步解释来源:")
        for item in mapping.explanation_evidence:
            if item.content:
                lines.append(f"  - [{item.source_type}] {item.source_name}｜{item.content}")
            else:
                lines.append(f"  - [{item.source_type}] {item.source_name}｜查询: {item.query}")
    return "\n".join(lines)


def mapping_to_dict(mapping: AttributeMappingResult) -> Dict[str, Any]:
    """把映射结果转成普通 dict，便于 API/debug/后续综合查询使用。"""
    return {
        "target": mapping.target,
        "label": mapping.label,
        "kg_field": mapping.kg_field,
        "values": mapping.values,
        "needs_explanation": mapping.needs_explanation,
        "note": mapping.note,
        "explanation_evidence": [
            {
                "source_type": item.source_type,
                "source_name": item.source_name,
                "query": item.query,
                "content": item.content,
            }
            for item in mapping.explanation_evidence
        ],
    }
