"""面向问答流程的 KG 查询服务。

kg_query 接收已经完成问题理解的 QueryPlan，自己根据计划调用
graph_store 的 Neo4j 数据访问函数。它不再从原始用户问题中重复提取主体。
"""

from dataclasses import dataclass
from typing import Optional

from neo4j import Driver

from .graph_store import (
    DangerousGoodFact,
    cache_dangerous_good_context,
    fact_to_dict,
    format_dangerous_good_fact,
    query_dangerous_good_fact,
)
from .query_understanding import QueryPlan


@dataclass(frozen=True)
class KGQueryResult:
    """KG 查询结果，兼容当前文本证据，也预留结构化事实。"""

    subject: str
    subject_type: Optional[str]
    context: Optional[str]
    fact: Optional[DangerousGoodFact] = None

    @property
    def found(self) -> bool:
        return self.fact is not None

    def fact_dict(self):
        return fact_to_dict(self.fact) if self.fact else None


def query_kg(driver: Optional[Driver], plan: QueryPlan) -> Optional[KGQueryResult]:
    """根据 QueryPlan 查询 KG。"""
    if not plan.requires_graph or not plan.graph_query:
        return None
    if driver is None:
        return None

    subject = plan.graph_query
    subject_type = plan.analysis.subject_type
    fact = query_dangerous_good_fact(driver, subject, subject_type)

    if fact is None:
        context = f"【图谱提示】在结构化图数据库中未检索到 {subject} 对应的官方危险货物条目。"
        cache_dangerous_good_context(subject, context)
    else:
        context = format_dangerous_good_fact(fact)
        cache_dangerous_good_context(subject, context)

    return KGQueryResult(
        subject=subject,
        subject_type=subject_type,
        context=context,
        fact=fact,
    )
