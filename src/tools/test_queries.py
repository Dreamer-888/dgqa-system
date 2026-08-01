#!/usr/bin/env python3
"""危险货物问答系统自动化回归测试脚本。

这个脚本直接复用 src/run.py 中的后端问答流程，不经过前端。

默认模式偏向“结构测试”：
- 禁用 L2 语义缓存，避免旧缓存掩盖真实问题；
- 禁用 LLM 调用，避免联网和费用；
- 主要检查路由、QueryPlan、KG 事实、综合映射、检索证据是否符合预期。

如需测试最终自然语言回答质量，可添加 --with-llm。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class TestCase:
    """单条回归测试用例。"""

    name: str
    question: str
    category: str
    expected_route: Optional[str] = None
    expected_target: Optional[str] = None
    expected_subject: Optional[str] = None
    expect_source_types: List[str] = field(default_factory=list)
    forbid_sources: List[str] = field(default_factory=list)
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    expected_cache_question: Optional[str] = None
    allow_llm_error_answer: bool = True


DEFAULT_CASES: List[TestCase] = [
    TestCase(
        name="综合-包装类别-UN",
        category="hybrid",
        question="UN1203的包装类别是什么？",
        expected_route="hybrid",
        expected_target="packing_group",
        expected_subject="UN1203",
        expect_source_types=["kg", "comprehensive_mapping", "text"],
        must_contain=["UN1203", "车用汽油或汽油", "Ⅱ", "包装组别", "具有中等危险性"],
    ),
    TestCase(
        name="综合-类别或项别-品名",
        category="hybrid",
        question="四氯乙烯属于哪一类？",
        expected_route="hybrid",
        expected_target="hazard_class",
        expected_subject="四氯乙烯",
        expect_source_types=["kg", "comprehensive_mapping", "text"],
        must_contain=["UN1897", "四氯乙烯", "6.1", "毒性物质"],
        must_not_contain=["所属危险主类别"],
    ),
    TestCase(
        name="综合-俗称解析",
        category="hybrid",
        question="酒精的包装类别是什么？",
        expected_route="hybrid",
        expected_target="packing_group",
        expected_subject="乙醇溶液",
        expect_source_types=["kg", "comprehensive_mapping", "text"],
        must_contain=["UN1170", "乙醇(酒精)或乙醇溶液", "Ⅱ", "Ⅲ", "包装"],
        must_not_contain=["UN2491", "乙醇胺"],
    ),
    TestCase(
        name="综合-包装类别解释",
        category="hybrid",
        question="为什么UN1212的包装类别是III？",
        expected_route="hybrid",
        expected_target="packing_group",
        expected_subject="UN1212",
        expect_source_types=["kg", "comprehensive_mapping"],
        must_contain=["UN1212", "Ⅲ", "GB 6944", "4.1.2"],
    ),
    TestCase(
        name="综合-类别原因",
        category="hybrid",
        question="为什么 UN1203 属于第3类？",
        expected_route="hybrid",
        expected_target="reason",
        expected_subject="UN1203",
        expect_source_types=["kg", "comprehensive_mapping"],
        must_contain=["UN1203", "第3类", "易燃液体", "5.3.1.1"],
        must_not_contain=["特殊规定290", "特殊规定362"],
    ),
    TestCase(
        name="综合-特殊规定索引",
        category="hybrid",
        question="UN1203的特殊规定是什么？",
        expected_route="hybrid",
        expected_target="special_provisions",
        expected_subject="UN1203",
        expected_cache_question="UN1203的特殊规定",
        expect_source_types=["kg", "comprehensive_mapping"],
        must_contain=["243", "火花点火式发动机"],
    ),
    TestCase(
        name="综合-特殊规定显式编号不一致",
        category="hybrid",
        question="UN1203的特殊规定290是什么意思？",
        expected_route="hybrid",
        expected_target="special_provisions",
        expected_subject="UN1203",
        expected_cache_question="UN1203的特殊规定",
        expect_source_types=["kg", "comprehensive_mapping"],
        must_contain=["特殊规定 290", "适用的特殊规定为 243", "特殊规定290"],
        must_not_contain=["特殊规定1203"],
    ),
    TestCase(
        name="直接-附录A精确编号",
        category="direct",
        question="特殊规定243是什么意思？",
        expected_route="direct",
        expected_target="special_provisions",
        expected_cache_question="特殊规定243的含义",
        expect_source_types=["comprehensive_mapping"],
        must_contain=["特殊规定243", "火花点火式发动机"],
        must_not_contain=["特殊规定290", "特殊规定362"],
    ),
    TestCase(
        name="直接-附录A不存在编号",
        category="direct",
        question="特殊规定999是什么意思？",
        expected_route="direct",
        expected_target="special_provisions",
        expect_source_types=["comprehensive_mapping"],
        must_contain=["特殊规定999", "未在GB 12268附录A精确索引中找到"],
        must_not_contain=["特殊规定243", "特殊规定290", "特殊规定362"],
    ),
    TestCase(
        name="综合-例外数量表",
        category="hybrid",
        question="UN1203的例外数量是多少？",
        expected_route="hybrid",
        expected_target="excepted_quantities",
        expected_subject="UN1203",
        expect_source_types=["kg", "comprehensive_mapping"],
        must_contain=["E2", "每件内包装最大净装载量", "30", "500"],
    ),
    TestCase(
        name="直接-GB6944定义",
        category="direct",
        question="GB6944中包装类别是什么？",
        expected_route="direct",
        expected_target="packing_group",
        expect_source_types=["text"],
        must_contain=["GB 6944", "包装类别"],
    ),
    TestCase(
        name="直接-GB6944隐式概念定义",
        category="direct",
        question="什么是易燃液体？",
        expected_route="direct",
        expected_target="definition",
        expected_cache_question="易燃液体的定义",
        expect_source_types=["text"],
        forbid_sources=["GB 12268", "附录A"],
        must_contain=["GB 6944", "5.3.1.1", "易燃液体"],
    ),
    TestCase(
        name="直接-裸标准号6944不进KG",
        category="direct",
        question="根据6944查询第3类易燃液体定义",
        expected_route="direct",
        expected_target="definition",
        expect_source_types=["text"],
        must_contain=["GB 6944", "第3类", "易燃液体"],
        must_not_contain=["UN6944", "Neo4j"],
    ),
    TestCase(
        name="直接-GB6944来源过滤",
        category="direct",
        question="GB6944中第3类易燃液体是怎么定义的？",
        expected_route="direct",
        expected_target="definition",
        expect_source_types=["text"],
        forbid_sources=["GB 12268", "附录A"],
        must_contain=["GB 6944", "第3类", "易燃液体"],
    ),
    TestCase(
        name="负例-不存在UN",
        category="negative",
        question="UN9999的包装类别是什么？",
        expected_route="direct",
        expected_target="packing_group",
        expect_source_types=["validation"],
        must_contain=["UN9999", "有效 UN 编号索引", "跳过 Neo4j 查询和文本模糊检索"],
    ),
]


def compact_text(value: Any, *, max_length: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= max_length else text[:max_length] + "..."


def collect_search_blob(response: Any, plan: Any) -> str:
    """把稳定证据合并为断言文本。"""
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    pieces = [
        plan.analysis.original,
        plan.analysis.normalized,
        plan.analysis.subject or "",
        plan.cache_question,
        payload.get("answer") or "",
        payload.get("prompt") or "",
        json.dumps(payload.get("sources") or [], ensure_ascii=False, default=str),
    ]
    return "\n".join(str(item) for item in pieces if item)


def source_types(response: Any) -> List[str]:
    sources = getattr(response, "sources", None)
    if sources is None and isinstance(response, dict):
        sources = response.get("sources", [])
    return sorted({str(item.get("type")) for item in sources or [] if item.get("type")})


def check_case(case: TestCase, response: Any, plan: Any) -> List[str]:
    failures: List[str] = []
    blob = collect_search_blob(response, plan)
    available_source_types = source_types(response)

    if case.expected_route and response.route != case.expected_route:
        failures.append(f"route 应为 {case.expected_route}，实际为 {response.route}")
    if case.expected_target and plan.analysis.target != case.expected_target:
        failures.append(f"target 应为 {case.expected_target}，实际为 {plan.analysis.target}")
    if case.expected_subject and plan.analysis.subject != case.expected_subject:
        failures.append(f"subject 应为 {case.expected_subject}，实际为 {plan.analysis.subject}")
    if case.expected_cache_question and plan.cache_question != case.expected_cache_question:
        failures.append(
            f"cache_question 应为 {case.expected_cache_question}，实际为 {plan.cache_question}"
        )

    for expected_type in case.expect_source_types:
        if expected_type not in available_source_types:
            failures.append(
                f"缺少证据类型 {expected_type}，实际证据类型为 {available_source_types}"
            )

    sources = getattr(response, "sources", None)
    if sources is None and isinstance(response, dict):
        sources = response.get("sources", [])
    for forbidden in case.forbid_sources:
        for source in sources or []:
            source_text = " ".join([
                str(source.get("source", "")),
                str(source.get("section_path", "")),
                str(source.get("title", "")),
                str(source.get("content", ""))[:300],
            ])
            if forbidden in source_text:
                failures.append(f"出现了禁止来源/内容：{forbidden}")
                break

    for text in case.must_contain:
        if text not in blob:
            failures.append(f"未找到期望文本：{text}")

    for text in case.must_not_contain:
        if text in blob:
            failures.append(f"出现了不应出现的文本：{text}")

    if not case.allow_llm_error_answer and str(response.answer).startswith("LLM 尚未配置"):
        failures.append("最终答案为 LLM 未配置提示")

    return failures


def render_markdown_report(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines = [
        "# 危险货物问答系统自动化测试报告",
        "",
        f"- 测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 总用例数：{summary['total']}",
        f"- 通过：{summary['passed']}",
        f"- 失败：{summary['failed']}",
        f"- LLM 模式：{'启用' if summary['with_llm'] else '禁用'}",
        f"- L2 缓存：{'启用' if summary['use_cache'] else '禁用'}",
        "",
        "| 结果 | 用例 | 类型 | 路由 | Target | Subject | 证据类型 | 耗时 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in results:
        status = "✅" if item["passed"] else "❌"
        lines.append(
            "| {status} | {name} | {category} | {route} | {target} | {subject} | {sources} | {elapsed:.2f}s |".format(
                status=status,
                name=item["name"],
                category=item["category"],
                route=item["route"],
                target=item["target"],
                subject=item["subject"] or "",
                sources=", ".join(item["source_types"]),
                elapsed=item["elapsed_seconds"],
            )
        )

    lines.append("")
    lines.append("## 失败详情")
    failed_items = [item for item in results if not item["passed"]]
    if not failed_items:
        lines.append("")
        lines.append("全部通过。")
    else:
        for item in failed_items:
            lines.extend([
                "",
                f"### {item['name']}",
                "",
                f"- 问题：{item['question']}",
                f"- 失败原因：{'；'.join(item['failures'])}",
                f"- 回答摘要：{item['answer_preview']}",
            ])

    lines.append("")
    lines.append("## 每条用例摘要")
    for item in results:
        lines.extend([
            "",
            f"### {item['name']}",
            "",
            f"- 问题：{item['question']}",
            f"- 路由：{item['route']}；Target：{item['target']}；Subject：{item['subject'] or '无'}",
            f"- 缓存：{item['cache_level'] or '未命中'}；LLM used：{item['llm_used']}",
            f"- 证据类型：{', '.join(item['source_types']) or '无'}",
            f"- 回答摘要：{item['answer_preview']}",
        ])

    return "\n".join(lines) + "\n"


def write_report(results: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"query_test_report_{stamp}.json"
    md_path = output_dir / f"query_test_report_{stamp}.md"

    json_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown_report(results, summary), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def run_tests(args: argparse.Namespace) -> int:
    import run as backend
    from graphrag import engine as retrieval

    if not args.with_llm:
        backend.QUERY_ANALYSIS_LLM_ENABLED = False
        backend.llm_engine.client = None

    if not args.use_cache:
        backend.L2_CACHE_ENABLED = False

    backend.L1_CACHE.clear()
    if args.clear_l2:
        backend.L2_CACHE.clear()
    if args.clear_l3:
        backend.L3_ENTITY_CACHE.clear()

    retrieval.init_engine(rebuild_index=args.rebuild_index)

    selected_cases = [
        case for case in DEFAULT_CASES
        if not args.category or case.category in args.category
    ]
    if args.limit:
        selected_cases = selected_cases[:args.limit]

    results: List[Dict[str, Any]] = []
    try:
        for index, case in enumerate(selected_cases, start=1):
            start = time.perf_counter()
            error: Optional[str] = None
            response = None
            plan = None
            failures: List[str] = []

            try:
                plan, analysis_llm_used = backend.analyze_question_plan(case.question)
                response = backend.ask(
                    backend.AskRequest(
                        question=case.question,
                        top_k=args.top_k,
                        return_prompt=True,
                    )
                )
                failures = check_case(case, response, plan)
            except Exception as exc:  # noqa: BLE001 - 测试脚本需要捕获并汇报所有异常
                error = repr(exc)
                failures = [f"执行异常：{error}"]
                analysis_llm_used = False

            elapsed = time.perf_counter() - start
            passed = not failures
            answer = getattr(response, "answer", "") if response is not None else ""
            result = {
                "index": index,
                "name": case.name,
                "category": case.category,
                "question": case.question,
                "passed": passed,
                "failures": failures,
                "elapsed_seconds": round(elapsed, 3),
                "route": getattr(response, "route", None) if response is not None else None,
                "target": getattr(plan.analysis, "target", None) if plan is not None else None,
                "subject": getattr(plan.analysis, "subject", None) if plan is not None else None,
                "cache_level": getattr(response, "cache_level", None) if response is not None else None,
                "from_cache": getattr(response, "from_cache", None) if response is not None else None,
                "llm_used": getattr(response, "llm_used", False) if response is not None else False,
                "analysis_llm_used": analysis_llm_used,
                "source_types": source_types(response) if response is not None else [],
                "answer_preview": compact_text(answer, max_length=240),
                "sources_preview": compact_text(
                    getattr(response, "sources", []) if response is not None else [],
                    max_length=600,
                ),
                "error": error,
            }
            results.append(result)

            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {index:02d}/{len(selected_cases)} {case.name} "
                f"route={result['route']} target={result['target']} "
                f"subject={result['subject']} elapsed={elapsed:.2f}s"
            )
            if failures:
                for failure in failures:
                    print(f"       - {failure}")

    finally:
        retrieval.close_engine()

    summary = {
        "total": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "with_llm": args.with_llm,
        "use_cache": args.use_cache,
        "top_k": args.top_k,
        "rebuild_index": args.rebuild_index,
    }
    paths = write_report(results, summary, PROJECT_ROOT / args.output_dir)
    print("")
    print(f"测试完成：{summary['passed']}/{summary['total']} 通过。")
    print(f"Markdown 报告：{paths['markdown']}")
    print(f"JSON 报告：{paths['json']}")
    return 0 if summary["failed"] == 0 else 1


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="危险货物问答系统自动化回归测试。")
    parser.add_argument("--with-llm", action="store_true", help="启用 LLM，测试最终回答质量。")
    parser.add_argument("--use-cache", action="store_true", help="启用 L2 语义缓存参与测试。")
    parser.add_argument("--clear-l2", action="store_true", help="测试前清空 L2 语义缓存。")
    parser.add_argument("--clear-l3", action="store_true", help="测试前清空 L3 Neo4j 实体缓存。")
    parser.add_argument("--rebuild-index", action="store_true", help="测试前重建文本向量索引。")
    parser.add_argument("--top-k", type=int, default=3, help="文本检索 top_k，默认 3。")
    parser.add_argument("--limit", type=int, default=0, help="只运行前 N 条用例。")
    parser.add_argument(
        "--category",
        action="append",
        choices=["kg", "hybrid", "direct", "negative"],
        help="只运行某类用例，可重复指定。",
    )
    parser.add_argument("--output-dir", default="reports", help="报告输出目录，默认 reports。")
    return parser.parse_args(argv)


def main() -> int:
    return run_tests(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
