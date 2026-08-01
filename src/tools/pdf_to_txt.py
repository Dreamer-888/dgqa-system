import fitz
import csv
import re
import os
from collections import Counter

PDF_PATH = "./data/source/GB+6944-2025.pdf"
OUTPUT_PATH = "./data/text/GB+6944.txt"
TABLE_OUTPUT_DIR = "./data/tables/GB6944"

start = 7
end = 27

def is_header(text):
    text = text.strip()
    if not text:
        return True
    if "GB 6944" in text or "GB6944" in text:
        return True
    if text == "危险货物分类和品名编号":
        return True
    return False

def is_footer(text):
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        return True
    if re.fullmatch(r"[—\-]\s*\d+\s*[—\-]", text):
        return True
    return False

def get_block_max_font_size(block):
    """取block中所有span的最大字号"""
    sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("text", "").strip():
                sizes.append(span["size"])
    return max(sizes) if sizes else 0.0

def get_block_raw_text(block):
    """从dict block拼出原始文本（行间用\\n）"""
    lines = []
    for line in block.get("lines", []):
        line_text = "".join(span["text"] for span in line.get("spans", []))
        lines.append(line_text)
    return "\n".join(lines)

def clean_block_text(text):
    """清洗排版换行，保证正文语意连贯"""
    # 先修复编号跨行断裂，如 "3.1\n2" → "3.12"，避免后续被当页码删掉
    text = re.sub(r'((?:\d+\.)+)\n(\d)', r'\1\2', text)
    # 逐行剔除纯数字行（页码混入块内部）
    lines = text.split("\n")
    lines = [l for l in lines if not re.fullmatch(r"\d+", l.strip())]
    text = "\n".join(lines)
    # 去掉排版换行
    text = text.replace("\n", "")
    # 合并多余空白
    text = re.sub(r"\s+", " ", text)
    # 中文字符之间的空格删除
    zh = r'[一-龥]'
    text = re.sub(fr'({zh})\s+({zh})', r'\1\2', text)
    # 修复错误标点组合
    text = text.replace("、。", "。")
    text = text.replace("，。", "。")
    text = text.replace("；。", "。")
    # 中文间的英文逗号转中文逗号
    text = re.sub(r'(?<=[一-龥]),(?=[一-龥])', '，', text)
    return text.strip()

def merge_blocks(block_items):
    """
    block_items: list of (text, is_title)
    标题单独成行；正文按语义连续拼接（上一段以句末标点结尾才换行）
    """
    result = []
    last_was_title = False

    for text, is_title in block_items:
        text = text.strip()
        if not text:
            continue
        if is_title:
            result.append(text)
            last_was_title = True
            continue
        # 正文：上一个是标题，或列表为空，则另起
        if not result or last_was_title:
            result.append(text)
            last_was_title = False
            continue
        # 正文：上一段以句末标点结尾则另起，否则拼接
        if result[-1].endswith(("。", "；", "：")):
            result.append(text)
        else:
            result[-1] += text
        last_was_title = False

    return "\n".join(result)

def is_title_by_pattern(text):
    """兜底：格式符合章节编号 + 中文且足够短，视为标题"""
    text = text.strip()
    if len(text) > 40:
        return False
    if text.endswith(("。", "；", "，", "、")):
        return False
    return bool(re.match(r'^\d+(\.\d+)*\s+[一-龥]', text))

def detect_body_font_size(doc):
    """扫描全部目标页，统计出现最多的字号作为正文字号"""
    sizes = []
    for page_num in range(start, end):
        page = doc[page_num]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(round(span["size"], 1))
    counter = Counter(sizes)
    return counter.most_common(1)[0][0] if counter else 10.0

def extract_pdf():
    if not os.path.exists(PDF_PATH):
        print("No such file in directory")
        return

    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
    doc = fitz.open(PDF_PATH)

    body_size = detect_body_font_size(doc)
    # 比正文字号大 0.5pt 以上视为标题
    title_threshold = body_size + 0.5

    all_items = []  # 跨所有页面的 (text, is_title) 列表
    table_counter = 0

    for page_num in range(start, end):
        page = doc[page_num]

        # ---------- 1. 提取表格 ----------
        tables = page.find_tables()
        table_infos = []
        if tables:
            for tbl in tables:
                bbox = tbl.bbox
                data = tbl.extract()
                clean_data = [row for row in data if any(cell and str(cell).strip() for cell in row)]
                if not clean_data:
                    continue
                csv_filename = f"page_{page_num}_table_{table_counter}.csv"
                csv_path = os.path.join(TABLE_OUTPUT_DIR, csv_filename)
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerows(clean_data)
                placeholder = f"[TABLE: {csv_filename}]"
                table_infos.append((bbox, placeholder))
                table_counter += 1

        table_infos.sort(key=lambda x: x[0][1])

        # ---------- 2. 提取文本块（带字体信息） ----------
        blocks = page.get_text("dict")["blocks"]
        blocks = sorted(blocks, key=lambda b: b["bbox"][1])

        table_idx = 0
        page_items = []  # list of (text, is_title)

        for block in blocks:
            if block["type"] != 0:
                continue

            x0, y0, x1, y1 = block["bbox"]
            raw_text = get_block_raw_text(block)
            if not raw_text.strip():
                continue
            if is_header(raw_text.strip()) or is_footer(raw_text.strip()):
                continue

            # 跳过表格内部文本
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            if any(tb[0] <= center_x <= tb[2] and tb[1] <= center_y <= tb[3] for tb, _ in table_infos):
                continue

            # 插入位于当前块之前的表格占位符
            while table_idx < len(table_infos):
                tb, placeholder = table_infos[table_idx]
                if tb[3] <= y0:
                    page_items.append((placeholder, True))
                    table_idx += 1
                else:
                    break

            cleaned = clean_block_text(raw_text)
            # 字号更大，或符合章节编号格式，视为标题
            is_title = (get_block_max_font_size(block) >= title_threshold
                        or is_title_by_pattern(cleaned))
            if cleaned:
                page_items.append((cleaned, is_title))

        # 处理页面末尾剩余的表格
        while table_idx < len(table_infos):
            _, placeholder = table_infos[table_idx]
            page_items.append((placeholder, True))
            table_idx += 1

        all_items.extend(page_items)

    final_text = merge_blocks(all_items)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"提取完成！共提取 {table_counter} 个表格，已保存至 {TABLE_OUTPUT_DIR}")
    print(f"文本内容已保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_pdf()
