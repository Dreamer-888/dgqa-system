"""Neo4j 图数据库访问层。

这个模块只负责根据已经确定的主体查询 Neo4j，并返回结构化事实。
它不负责理解用户问题，也不负责决定答案怎么生成。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import Driver

from .cache import L3_ENTITY_CACHE
from .definitions import CLASS_NAMES, DIVISION_NAMES


@dataclass(frozen=True)
class DangerousGoodFact:
    """GB 12268 图谱中的危险货物结构化事实。"""

    subject: str
    un_number: str
    name_zh: str
    name_en: str
    class_or_divisions: List[str]
    class_or_division_labels: List[str]
    hazard_class: str
    hazard_class_id: str
    subsidiary_hazard: str
    packing_groups: List[str]
    special_provisions: str
    limited_quantities: str
    excepted_quantities: str
    packing_instruction: str
    portable_tank_special_provisions: str


def _clean_value(value: Any, default: str = "无") -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return default
    return text


def _normalize_subject(subject: str) -> str:
    return subject.strip().upper() if subject.strip().upper().startswith("UN") else subject.strip()


def _list_clean_values(values: Any) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    result = []
    seen = set()
    for value in values:
        text = _clean_value(value, "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _class_or_division_label(value: str) -> str:
    name = DIVISION_NAMES.get(value) or CLASS_NAMES.get(value)
    return f"{value} {name}" if name else value


def format_dangerous_good_fact(fact: DangerousGoodFact) -> str:
    """将结构化事实格式化为现有 prompt/answer 可读的 KG 证据文本。"""
    group_text = " / ".join(fact.packing_groups) if fact.packing_groups else "（标准未明确指定包装组）"
    class_or_division_text = (
        " / ".join(fact.class_or_division_labels)
        if fact.class_or_division_labels
        else fact.hazard_class or "无"
    )
    return (
        "【图谱高精度事实数据｜信息源: GB 12268 Neo4j 结构化库】\n"
        f"  - 联合国编号 (UN号): {fact.un_number or '无'}\n"
        f"  - 中文正式名称: {fact.name_zh or '无'}\n"
        f"  - 英文正式名称: {fact.name_en or '无'}\n"
        f"  - 类别或项别: {class_or_division_text}\n"
        f"  - 所属危险主类: {fact.hazard_class or '无'} "
        f"(主类编号: {fact.hazard_class_id or '无'})\n"
        f"  - 次要危险性: {fact.subsidiary_hazard or '无次要危险性'}\n"
        f"  - 允许使用的所有包装组别: {group_text}\n"
        f"  - 特殊规定条文索引: {fact.special_provisions or '无'}\n"
        f"  - 有限数量限制: {fact.limited_quantities or '无'}\n"
        f"  - 例外数量编码: {fact.excepted_quantities or '无'}\n"
        f"  - 包装规范指令: {fact.packing_instruction or '无'}\n"
        f"  - 移动罐体特殊规定: {fact.portable_tank_special_provisions or '无'}\n"
    )


def cache_dangerous_good_context(subject: str, content: str) -> None:
    """写入 L3 KG 证据文本缓存。"""
    normalized_subject = _normalize_subject(subject)
    if normalized_subject and content:
        L3_ENTITY_CACHE.put(normalized_subject, content)


def _fact_from_record(subject: str, record: Any) -> DangerousGoodFact:
    dangerous_good = record["d"]
    hazard_class = record["h"]
    relation = record["r"]
    class_or_divisions = _list_clean_values(
        relation.get("class_or_divisions") or dangerous_good.get("class_or_divisions")
    )
    groups = sorted({
        str(group)
        for group in record["packing_groups"]
        if group and str(group).lower() != "nan"
    })
    subsidiary = _clean_value(relation.get("subsidiary_hazard"), "无次要危险性")

    return DangerousGoodFact(
        subject=subject,
        un_number=_clean_value(dangerous_good.get("un_number"), ""),
        name_zh=_clean_value(dangerous_good.get("name_zh"), ""),
        name_en=_clean_value(dangerous_good.get("name_en"), ""),
        class_or_divisions=class_or_divisions,
        class_or_division_labels=[
            _class_or_division_label(value) for value in class_or_divisions
        ],
        hazard_class=_clean_value(hazard_class.get("class_name"), ""),
        hazard_class_id=_clean_value(hazard_class.get("class_id"), ""),
        subsidiary_hazard=subsidiary,
        packing_groups=groups,
        special_provisions=_clean_value(dangerous_good.get("special_provisions"), ""),
        limited_quantities=_clean_value(dangerous_good.get("limited_quantities"), ""),
        excepted_quantities=_clean_value(dangerous_good.get("excepted_quantities"), ""),
        packing_instruction=_clean_value(dangerous_good.get("packing_instruction"), ""),
        portable_tank_special_provisions=_clean_value(
            dangerous_good.get("portable_tank_special_provisions"), ""
        ),
    )


def query_dangerous_good_fact(
    driver: Optional[Driver],
    subject: str,
    subject_type: Optional[str] = None,
) -> Optional[DangerousGoodFact]:
    """按已分析出的主体查询 Neo4j，返回结构化危险货物事实。"""
    normalized_subject = _normalize_subject(subject)
    if not normalized_subject or driver is None:
        return None

    if subject_type == "un_number" or normalized_subject.startswith("UN"):
        print(f"\n[Neo4j 查询] UN 编号实体: {normalized_subject}")
        cypher = """
        MATCH (d:DangerousGood {un_number: $un_number})-[r:BELONGS_TO]->(h:HazardClass)
        OPTIONAL MATCH (d)-[:REQUIRES_PACKING]->(p:PackingGroup)
        RETURN d, r, h, collect(p.group_rating) AS packing_groups
        LIMIT 1
        """
        params = {"un_number": normalized_subject}
    else:
        print(f"\n[Neo4j 查询] 品名实体: {normalized_subject}")
        cypher = """
        MATCH (d:DangerousGood)-[r:BELONGS_TO]->(h:HazardClass)
        WHERE d.name_zh = $entity_name
           OR $entity_name IN coalesce(d.name_zh_aliases, [])
           OR d.name_zh STARTS WITH $entity_name
           OR d.name_zh CONTAINS $entity_name
        WITH d, r, h,
             CASE
                 WHEN d.name_zh = $entity_name THEN 0
                 WHEN $entity_name IN coalesce(d.name_zh_aliases, []) THEN 1
                 WHEN d.name_zh STARTS WITH $entity_name THEN 2
                 ELSE 3
             END AS match_rank
        OPTIONAL MATCH (d)-[:REQUIRES_PACKING]->(p:PackingGroup)
        RETURN d, r, h, collect(p.group_rating) AS packing_groups, match_rank
        ORDER BY match_rank ASC, size(d.name_zh) ASC, d.un_number ASC
        LIMIT 1
        """
        params = {"entity_name": normalized_subject}

    try:
        with driver.session() as session:
            record = session.run(cypher, **params).single()
            if not record:
                return None
            return _fact_from_record(normalized_subject, record)
    except Exception as exc:
        print(f"Neo4j 图谱查询异常: {exc}，系统将退化为文本检索。")
        return None


def query_dangerous_good_context(
    driver: Optional[Driver],
    subject: str,
    subject_type: Optional[str] = None,
) -> Optional[str]:
    """按主体查询并返回 KG 证据文本，带 L3 实体缓存。"""
    normalized_subject = _normalize_subject(subject)
    if not normalized_subject:
        return None

    cached = L3_ENTITY_CACHE.get(normalized_subject)
    if cached is not None:
        print(f"[L3 实体缓存命中] {normalized_subject}")
        return cached

    fact = query_dangerous_good_fact(driver, normalized_subject, subject_type)
    if fact is None:
        content = f"【图谱提示】在结构化图数据库中未检索到 {normalized_subject} 对应的官方危险货物条目。"
    else:
        content = format_dangerous_good_fact(fact)

    L3_ENTITY_CACHE.put(normalized_subject, content)
    return content


def fact_to_dict(fact: DangerousGoodFact) -> Dict[str, Any]:
    """用于 API/debug 的结构化事实字典。"""
    return {
        "subject": fact.subject,
        "un_number": fact.un_number,
        "name_zh": fact.name_zh,
        "name_en": fact.name_en,
        "class_or_divisions": fact.class_or_divisions,
        "class_or_division_labels": fact.class_or_division_labels,
        "hazard_class": fact.hazard_class,
        "hazard_class_id": fact.hazard_class_id,
        "subsidiary_hazard": fact.subsidiary_hazard,
        "packing_groups": fact.packing_groups,
        "special_provisions": fact.special_provisions,
        "limited_quantities": fact.limited_quantities,
        "excepted_quantities": fact.excepted_quantities,
        "packing_instruction": fact.packing_instruction,
        "portable_tank_special_provisions": fact.portable_tank_special_provisions,
    }
