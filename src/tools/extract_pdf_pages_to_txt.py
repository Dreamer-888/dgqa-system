"""按页码范围从 PDF 提取正文到 txt。

示例：
    ./venv/bin/python src/tools/extract_pdf_pages_to_txt.py \
        --pdf data/source/GB+12268-2025.pdf \
        --start-page 389 \
        --end-page 412 \
        --output data/text/GB+12268附录389-412.txt

页码使用常见的 1-based 页码；脚本内部会自动转换为 PyMuPDF 的 0-based 索引。
输出 txt 只包含提取正文，不额外写入【PDF第xxx页】这类页码标识。
"""

import argparse
import os
import re
from pathlib import Path

import fitz


def is_noise_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d+", stripped):
        return True
    if re.fullmatch(r"[—\-]\s*\d+\s*[—\-]", stripped):
        return True
    if "GB 12268" in stripped or "GB12268" in stripped:
        return True
    return False


def clean_line(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    zh = r"[一-龥]"
    text = re.sub(fr"({zh})\s+({zh})", r"\1\2", text)
    text = re.sub(r"(?<=[一-龥]),(?=[一-龥])", "，", text)
    return text.strip()


def merge_lines(lines: list[str]) -> str:
    merged: list[str] = []
    for line in lines:
        line = clean_line(line)
        if not line or is_noise_line(line):
            continue
        if not merged:
            merged.append(line)
            continue
        if re.match(r"^[A-Z]?\d+(?:\.\d+)*\s+", line):
            merged.append(line)
        elif merged[-1].endswith(("。", "；", "：", "）", ")", "。")):
            merged.append(line)
        else:
            merged[-1] += line
    return "\n".join(merged)


def extract_pages(pdf_path: Path, start_page: int, end_page: int, output_path: Path) -> None:
    if start_page < 1 or end_page < start_page:
        raise ValueError("页码范围不合法：请保证 1 <= start_page <= end_page")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        if end_page > doc.page_count:
            raise ValueError(f"PDF 只有 {doc.page_count} 页，不能提取到第 {end_page} 页。")

        page_texts: list[str] = []
        for page_no in range(start_page, end_page + 1):
            page = doc[page_no - 1]
            raw_text = page.get_text("text")
            lines = raw_text.splitlines()
            text = merge_lines(lines)
            if text:
                page_texts.append(text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n\n".join(page_texts), encoding="utf-8")
    finally:
        doc.close()

    print(f"提取完成：PDF 第 {start_page}-{end_page} 页")
    print(f"输出文件：{output_path}")
    print(f"文件大小：{os.path.getsize(output_path)} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 PDF 指定页码范围提取文本到 txt。")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径。")
    parser.add_argument("--start-page", type=int, required=True, help="起始页码，1-based。")
    parser.add_argument("--end-page", type=int, required=True, help="结束页码，1-based，包含该页。")
    parser.add_argument("--output", required=True, help="输出 txt 文件路径。")
    args = parser.parse_args()

    extract_pages(
        pdf_path=Path(args.pdf),
        start_page=args.start_page,
        end_page=args.end_page,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
