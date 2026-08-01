"""单独重建文本检索索引。

只处理 data/text/*.txt，生成：
- data/text/faiss_index.bin
- data/text/metadata.json

不会启动 FastAPI，也不会连接 Neo4j。

推荐从项目根目录运行：
    python3 ./src/tools/rebuild_text_index.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphrag.settings import apply_runtime_environment

apply_runtime_environment()

from sentence_transformers import SentenceTransformer

from graphrag import engine
from graphrag.definitions import Config
from graphrag.chunker import (
    attach_table_metadata,
    discover_text_files,
    load_all_tables,
    parse_all_text_files,
)


def rebuild_text_index(dry_run: bool = False) -> None:
    text_files = discover_text_files(Config.TEXT_DIR)
    if not text_files:
        raise RuntimeError(f"{Config.TEXT_DIR} 下没有可用于索引的 .txt 文件。")

    print("将要索引的文本文件：")
    for path in text_files:
        print(f"  - {path}")

    tables = load_all_tables(Config.TABLE_DIR)
    chunks = parse_all_text_files(
        Config.TEXT_DIR,
        Config.TEXT_FILE,
        Config.CHUNK_MAX_LENGTH,
    )
    chunks = attach_table_metadata(chunks, tables)

    print(f"文本分块数量：{len(chunks)}")
    if dry_run:
        print("dry-run 模式：仅检查文本解析，不生成索引文件。")
        return

    print(f"正在加载嵌入模型：{Config.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    index = engine.build_indices(chunks, embedding_model)
    engine.save_index_and_metadata(
        index,
        engine.id_to_metadata,
        Config.INDEX_SAVE_PATH,
        Config.METADATA_SAVE_PATH,
    )
    print("文本检索索引重建完成：")
    print(f"  - {Config.INDEX_SAVE_PATH}")
    print(f"  - {Config.METADATA_SAVE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="重建 RAG 文本检索 FAISS 索引和 metadata。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 data/text/*.txt 解析结果，不生成索引。",
    )
    args = parser.parse_args()
    rebuild_text_index(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
