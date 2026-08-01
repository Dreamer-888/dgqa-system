"""文本分块与文本源准备。

这个模块只负责把 data/text/*.txt 解析成可入库的 chunk，并处理
GB6944 文本中的 [TABLE: xxx] 表格占位符。

FAISS/BM25/reranker 检索由 text_search.py 负责，证据组装由 evidence.py
负责，engine.py 保留引擎生命周期和 GraphRAG 查询编排入口。
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .definitions import Config, TEXT_DEFINITION_MARKERS


def load_all_tables(table_dir: str) -> Dict[str, pd.DataFrame]:
    tables = {}
    if not os.path.exists(table_dir):
        print(f"Warning: No such directory {table_dir}")
        return tables

    for filename in os.listdir(table_dir):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(table_dir, filename)
            try:
                df = pd.read_csv(filepath)
                tables[filename] = df
                print(f"Loading table: {filename}")
            except Exception as e:
                print(f"Warning: table {filename} load failed: {e}")
    return tables


def normalize_title_number(text: str) -> str:
    """修复 4. 1 / 4 . 1 / 5.1. 1 这类标题编号空格。"""
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s+\.\s*(?=\d)", ".", text)
    return text.strip()


def split_long_text_with_overlap(text: str, max_len: int, overlap: int) -> List[str]:
    """长文本兜底切分，尽量按句号/分号/编号项切。"""
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    parts = re.split(
        r"(?<=[。；;])|(?=[a-zA-Z][)）]\s*)|(?=\([0-9]+\)\s*)|(?=\d+[)）]\s*)",
        text,
    )
    chunks = []
    buf = ""

    for part in parts:
        if not part.strip():
            continue
        if len(buf) + len(part) > max_len and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap:] + part
        else:
            buf += part

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def is_structural_section_title(line: str) -> Tuple[Optional[str], int]:
    """识别 4.1、4.1.2、5.9.4 这类章节号。"""
    line = normalize_title_number(line)
    match = re.match(r"^(\d{1,2}(?:\.\d+)+)\s+\S+", line)
    if not match:
        return None, 0
    section_id = match.group(1)
    first_number = int(section_id.split(".", 1)[0])
    if first_number < 1 or first_number > 20:
        return None, 0
    rest = line[len(section_id):].strip()
    if rest.startswith(("项", "和", "及", "或", "的", "中", "(", "（")):
        return None, 0
    return section_id, section_id.count(".") + 1


def is_structural_numbered_item(line: str) -> Optional[str]:
    """识别 16、144、378 等纯数字条目号。"""
    match = re.match(r"^(\d{2,4})\s+\S+", line.strip())
    return match.group(1) if match else None


def is_structural_sub_item(line: str) -> bool:
    """识别 a)、b）、1)、2）等小节号；小节永远附属于上一块。"""
    return re.match(r"^(?:[a-zA-Z]|\d+)[)）]\s*", line.strip()) is not None


def is_structural_note(line: str) -> bool:
    """识别 注、注1、注2：等说明段。"""
    return re.match(r"^注\d*[:：\s]", line.strip()) is not None


def is_definition_text(text: str) -> bool:
    return any(marker in text for marker in TEXT_DEFINITION_MARKERS)


def normalize_structural_line(line: str) -> str:
    """清洗单行文本，保留结构编号。"""
    line = normalize_title_number(line.strip())
    line = re.sub(r"\s+", " ", line)
    zh = r"[一-龥]"
    line = re.sub(fr"({zh})\s+({zh})", r"\1\2", line)
    return line.strip()


def infer_document_title(text_file: str, source_name: str, lines: List[str]) -> str:
    """给没有显式章节路径的编号条目一个稳定的上级路径。"""
    for line in lines[:20]:
        if line.startswith("附录"):
            return line
    filename = os.path.splitext(os.path.basename(text_file))[0]
    if "附录" in filename:
        return filename
    return source_name


def display_title_for_structural_line(line: str, section_id: str) -> str:
    """给章节路径使用的短标题。"""
    if len(line) > 80 or line.endswith(("。", "；", ";")):
        return section_id
    return line


def make_chunk(
    *,
    source_name: str,
    section_id: str,
    section_path: str,
    title: str,
    text: str,
    level: int,
    is_definition: bool,
) -> Dict:
    prefix = "【核心定义】" if is_definition else ""
    return {
        "source": source_name,
        "section_id": section_id,
        "section_path": section_path,
        "title": title,
        "text": text,
        "vector_input": (
            f"{prefix}来源：{source_name}。"
            f"章节路径：{section_path}。"
            f"正文：{text}"
        ),
        "level": level,
        "is_definition": is_definition,
    }


def build_structural_chunks(
    text_file: str,
    lines: List[str],
    max_len: int,
    source_name: str,
) -> List[Dict]:
    """通用结构化分块器。"""
    document_title = infer_document_title(text_file, source_name, lines)
    section_stack: Dict[int, str] = {}
    chunks: List[Dict] = []

    current_title = "前言/背景"
    current_section_id = ""
    current_level = 0
    current_path = document_title
    current_lines: List[str] = []

    def section_path_for_stack() -> str:
        if not section_stack:
            return document_title
        return " / ".join(section_stack[i] for i in sorted(section_stack.keys()))

    def flush_current() -> None:
        nonlocal current_lines
        content = " ".join(line.strip() for line in current_lines if line.strip())
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            current_lines = []
            return

        has_def = is_definition_text(content)
        parts = split_long_text_with_overlap(content, max_len, Config.CHUNK_OVERLAP)
        for index, part in enumerate(parts, start=1):
            title = current_title
            text = part
            if index > 1:
                title = f"{current_title}（续{index - 1}）"
                if not text.startswith(current_section_id):
                    text = f"{current_title}（续）：{text}"
            chunks.append(make_chunk(
                source_name=source_name,
                section_id=current_section_id,
                section_path=current_path,
                title=title,
                text=text,
                level=current_level,
                is_definition=has_def,
            ))
        current_lines = []

    for raw_line in lines:
        line = normalize_structural_line(raw_line)
        if not line:
            continue

        is_child_line = is_structural_sub_item(line) or is_structural_note(line)
        section_id, section_level = (None, 0) if is_child_line else is_structural_section_title(line)
        numbered_id = None if is_child_line else is_structural_numbered_item(line)

        if section_id:
            flush_current()
            for level in list(section_stack.keys()):
                if level >= section_level:
                    del section_stack[level]
            display_title = display_title_for_structural_line(line, section_id)
            section_stack[section_level] = display_title
            current_title = display_title
            current_section_id = section_id
            current_level = section_level
            current_path = section_path_for_stack()
            current_lines = [line]
            continue

        if numbered_id:
            flush_current()
            current_title = numbered_id
            current_section_id = numbered_id
            current_level = 1
            base_path = document_title
            current_path = f"{base_path} / {numbered_id}" if base_path else numbered_id
            current_lines = [line]
            continue

        if not current_lines:
            current_title = "前言/背景"
            current_section_id = ""
            current_level = 0
            current_path = document_title
        current_lines.append(line)

        if not current_section_id and len(" ".join(current_lines)) >= max_len:
            flush_current()

    flush_current()
    print(f" 通用结构化分块完成！共生成 {len(chunks)} 个语义块。")
    return chunks


def source_name_from_text_file(text_file: str) -> str:
    """根据 txt 文件名生成检索来源名称，便于多标准文本混合入库后溯源。"""
    filename = os.path.basename(text_file)
    stem = os.path.splitext(filename)[0]
    if "6944" in stem:
        return "GB 6944-2025"
    if "12268" in stem:
        return "GB 12268-2025"
    return stem


def discover_text_files(text_dir: str) -> List[str]:
    """读取 data/text 目录下所有 txt 文件，作为 RAG 文本知识库来源。"""
    if not os.path.isdir(text_dir):
        return []
    files = []
    for filename in sorted(os.listdir(text_dir)):
        filepath = os.path.join(text_dir, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(".txt"):
            files.append(filepath)
    return files


def parse_all_text_files(
    text_dir: str = Config.TEXT_DIR,
    legacy_text_file: str = Config.TEXT_FILE,
    max_len: int = Config.CHUNK_MAX_LENGTH,
) -> List[Dict]:
    """优先加载 data/text/*.txt；若目录为空，则回退到旧的单文件路径。"""
    text_files = discover_text_files(text_dir)
    if not text_files and os.path.exists(legacy_text_file):
        print(f"Warning: {text_dir} 下未发现 txt 文件，回退加载旧文本: {legacy_text_file}")
        text_files = [legacy_text_file]

    chunks: List[Dict] = []
    for text_file in text_files:
        source_name = source_name_from_text_file(text_file)
        print(f"Loading text source: {text_file} ({source_name})")
        chunks.extend(parse_and_chunk_text(text_file, max_len, source_name=source_name))

    if not chunks:
        print(f"Warning: 未加载到任何 txt 文本，请检查 {text_dir}")
    else:
        print(f" 多文本知识库分块完成！共加载 {len(text_files)} 个 txt，生成 {len(chunks)} 个语义块。")
    return chunks


def parse_and_chunk_text(
    text_file: str,
    max_len: int = Config.CHUNK_MAX_LENGTH,
    source_name: Optional[str] = None,
) -> List[Dict]:
    source_name = source_name or source_name_from_text_file(text_file)
    with open(text_file, "r", encoding="utf-8") as f:
        raw_lines = []
        for line in f.readlines():
            normalized = normalize_structural_line(line)
            if normalized:
                raw_lines.append(normalized)

    return build_structural_chunks(text_file, raw_lines, max_len, source_name)


def attach_table_metadata(chunks: List[Dict], tables: Dict[str, pd.DataFrame]) -> List[Dict]:
    """处理文本 chunk 中的 [TABLE: xxx] 占位符。"""
    for chunk in chunks:
        text = chunk["text"]
        refs = re.findall(r"\[TABLE:\s*(.*?)\]", text)
        table_markdowns = []
        for ref in refs:
            if ref in tables:
                df = tables[ref]
                try:
                    md = df.to_markdown(index=False)
                except ImportError:
                    md = df.to_csv(index=False)
                table_markdowns.append(f"【表格引用: {ref}】\n{md}")
            else:
                table_markdowns.append(f"【警告】未找到表格文件: {ref}")

        chunk["metadata"] = {
            "attached_tables": "\n\n".join(table_markdowns) if table_markdowns else "",
            "raw_refs": refs,
        }
    return chunks
