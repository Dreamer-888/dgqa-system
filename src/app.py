from typing import Any, Dict, List, Optional

import httpx
import streamlit as st


st.set_page_config(
    page_title="危险货物智能问答系统",
    page_icon="🧪",
    layout="wide",
)

ROUTE_NAMES = {
    "kg": "知识图谱",
    "direct": "直接查询",
    "hybrid": "综合查询",
}

EXAMPLE_QUESTIONS = [
    "UN1203 的包装组是什么？",
    "汽油属于哪一类危险货物？",
    "什么是易燃液体？",
    "为什么 UN1203 属于第3类？",
]


def call_api(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """调用 run.py 暴露的 FastAPI 接口，并统一处理响应。"""
    timeout = httpx.Timeout(connect=5, read=180, write=10, pool=5)
    url = f"{base_url.rstrip('/')}{path}"
    with httpx.Client(timeout=timeout) as client:
        response = client.request(method, url, json=payload)
        response.raise_for_status()
        return response.json()


def error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "无法连接 FastAPI，请确认后端已经启动，并检查后端地址。"
    if isinstance(exc, httpx.TimeoutException):
        return "后端响应超时。首次加载模型可能较慢，请稍后重试。"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
            detail = body.get("detail", body)
        except ValueError:
            detail = exc.response.text
        return f"后端返回 {exc.response.status_code}：{detail}"
    return f"请求失败：{exc}"


def render_sources(sources: List[Dict[str, Any]], key_prefix: str) -> None:
    if not sources:
        st.info("本次请求没有返回检索证据。")
        return

    for index, source in enumerate(sources, start=1):
        source_type = "知识图谱" if source.get("type") == "kg" else "法规文本"
        source_name = source.get("source", "未知来源")
        section = source.get("section_path") or source.get("title") or ""
        label = f"{index}. {source_type}｜{source_name}"
        if section:
            label += f"｜{section}"

        with st.expander(label):
            if source.get("section_path"):
                st.caption(f"章节路径：{source['section_path']}")
            st.text(source.get("content") or "无正文内容")


def render_answer(result: Dict[str, Any], message_index: int) -> None:
    st.markdown(result.get("answer") or "后端未返回回答。")

    route = ROUTE_NAMES.get(result.get("route"), result.get("route", "未知"))
    source_count = len(result.get("sources", []))
    llm_status = "已调用" if result.get("llm_used") else "未调用"
    if result.get("from_cache"):
        cache_status = f"命中 {result.get('cache_level') or ''}".strip()
    else:
        cache_status = "未命中"

    st.caption(
        f"路由：{route}　·　证据：{source_count} 条　·　"
        f"大模型：{llm_status}　·　缓存：{cache_status}"
    )
    if result.get("cache_level") == "L2":
        verification = (
            " · 已由LLM复核"
            if result.get("cache_verified_by_llm")
            else " · 复核状态未知"
        )
        st.caption(
            f"语义匹配问题：{result.get('matched_question', '未知')}　·　"
            f"相似度：{result.get('semantic_similarity', 0):.4f}{verification}"
        )

    with st.expander("查看检索依据"):
        render_sources(result.get("sources", []), f"message-{message_index}")

    if result.get("prompt"):
        with st.expander("查看发送给大模型的 Prompt"):
            st.code(result["prompt"], language="text")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "prompt_result" not in st.session_state:
    st.session_state.prompt_result = None


with st.sidebar:
    st.title("系统控制台")
    api_url = st.text_input(
        "FastAPI 地址",
        value="http://localhost:8000",
        help="对应 run.py 启动后的服务地址。",
    )
    top_k = st.slider(
        "文本召回数量",
        min_value=1,
        max_value=8,
        value=3,
        help="对应 AskRequest.top_k；KG 精确查询不受此参数影响。",
    )
    return_prompt = st.checkbox(
        "问答时返回 Prompt",
        value=False,
        help="对应 AskRequest.return_prompt。",
    )

    status_area = st.empty()
    if st.button("检查后端状态", use_container_width=True):
        try:
            health = call_api(api_url, "GET", "/health")
            if health.get("status") == "ok":
                cache = health.get("cache", {})
                l1 = cache.get("l1_answer", {})
                l2 = cache.get("l2_semantic", {})
                l3 = cache.get("l3_entity", {})
                status_area.success(
                    "检索引擎："
                    f"{health.get('retrieval_engine', 'unknown')}\n\n"
                    "大模型："
                    f"{health.get('llm_model') if health.get('llm_enabled') else '未启用'}\n\n"
                    f"缓存：L1 {l1.get('size', 0)}/{l1.get('maxsize', 0)} · "
                    f"L2 {l2.get('size', 0)}/{l2.get('maxsize', 0)} · "
                    f"L3 {l3.get('size', 0)}/{l3.get('maxsize', 0)}"
                )
            else:
                status_area.warning(str(health))
        except (httpx.HTTPError, ValueError) as exc:
            status_area.error(error_message(exc))

    st.divider()
    st.caption("快速测试")
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example-{index}", use_container_width=True):
            st.session_state.pending_question = example

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


st.title("危险货物智能问答系统")
st.caption("Neo4j 知识图谱 · FAISS + BM25 混合检索 · Reranker · LLM")

chat_tab, debug_tab = st.tabs(["智能问答", "检索调试"])


with chat_tab:
    if not st.session_state.messages:
        st.info(
            "输入危险货物法规问题。事实属性问题优先查询 Neo4j，"
            "定义与规则问题检索 GB 6944 文本，复杂问题采用混合检索。"
        )

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message.get("error"):
                    st.error(message["error"])
                else:
                    render_answer(message["result"], message_index)

    typed_question = st.chat_input("例如：UN1203 的包装组是什么？")
    question = typed_question or st.session_state.pending_question

    if question:
        st.session_state.pending_question = None
        cleaned_question = " ".join(question.strip().split())
        if cleaned_question:
            st.session_state.messages.append(
                {"role": "user", "content": cleaned_question}
            )
            try:
                with st.spinner("正在检索知识图谱与法规文本……"):
                    answer_result = call_api(
                        api_url,
                        "POST",
                        "/ask",
                        {
                            "question": cleaned_question,
                            "top_k": top_k,
                            "return_prompt": return_prompt,
                        },
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "result": answer_result}
                )
            except (httpx.HTTPError, ValueError) as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "error": error_message(exc)}
                )
            st.rerun()


with debug_tab:
    st.subheader("仅检索，不调用大模型")
    st.caption(
        "这里直接调用 run.py 的 /prompt 接口，适合检查问题路由、"
        "Neo4j 命中结果、文本召回内容以及最终 Prompt。"
    )

    with st.form("prompt-debug-form"):
        debug_question = st.text_area(
            "测试问题",
            height=90,
            placeholder="例如：为什么 UN1203 属于第3类？",
        )
        debug_submitted = st.form_submit_button(
            "生成检索 Prompt",
            type="primary",
            use_container_width=True,
        )

    if debug_submitted:
        if not debug_question.strip():
            st.warning("请输入测试问题。")
        else:
            try:
                with st.spinner("正在执行检索……"):
                    st.session_state.prompt_result = call_api(
                        api_url,
                        "POST",
                        "/prompt",
                        {
                            "question": debug_question,
                            "top_k": top_k,
                            "return_prompt": False,
                        },
                    )
            except (httpx.HTTPError, ValueError) as exc:
                st.session_state.prompt_result = {"_error": error_message(exc)}

    prompt_result = st.session_state.prompt_result
    if prompt_result:
        if prompt_result.get("_error"):
            st.error(prompt_result["_error"])
        else:
            route = ROUTE_NAMES.get(
                prompt_result.get("route"),
                prompt_result.get("route", "未知"),
            )
            col1, col2 = st.columns(2)
            col1.metric("问题路由", route)
            col2.metric("证据数量", len(prompt_result.get("sources", [])))

            st.markdown("#### 检索证据")
            render_sources(prompt_result.get("sources", []), "debug")

            st.markdown("#### 最终 Prompt")
            st.code(prompt_result.get("prompt", ""), language="text")
