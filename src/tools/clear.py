#!/usr/bin/env python3
"""缓存清理工具。

默认清理会长期影响测试结果的持久化缓存：
- L2 语义问答缓存：data/cache/semantic_cache.db
- 别名记忆缓存：data/cache/alias_memory.db

说明：
- L1 是后端进程内的短期精确问答缓存，重启后端即可清空。
- L3 是后端进程内的 Neo4j 实体缓存；本脚本只能清理当前脚本进程内的 L3，
  不能清掉已经运行中的后端进程内存。若要完全清空 L1/L3，请重启后端。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphrag.settings import settings


def default_l2_path() -> Path:
    return settings.cache.l2_db


def default_alias_path() -> Path:
    return settings.cache.alias_memory_db


def clear_l2(path: Path) -> None:
    from graphrag.cache import L2SemanticCache

    cache = L2SemanticCache(database_path=path)
    before = cache.stats()
    cache.clear()
    after = cache.stats()
    print(f"✅ L2 语义缓存已清理：{path}")
    print(f"   size: {before['size']} -> {after['size']}")


def clear_alias(path: Path) -> None:
    from graphrag.cache import AliasMemoryCache

    cache = AliasMemoryCache(database_path=path)
    before = cache.stats()
    cache.clear()
    after = cache.stats()
    print(f"✅ 别名记忆缓存已清理：{path}")
    print(f"   size: {before['size']} -> {after['size']}")


def clear_l3_memory() -> None:
    from graphrag.cache import L3_ENTITY_CACHE

    before = L3_ENTITY_CACHE.stats()
    L3_ENTITY_CACHE.clear()
    after = L3_ENTITY_CACHE.stats()
    print("✅ 当前脚本进程内的 L3 Neo4j 实体缓存已清理")
    print(f"   size: {before['size']} -> {after['size']}")
    print("   提醒：如果后端服务正在运行，它自己的 L3 内存缓存需要重启后端才会清空。")


def show_stats(l2_path: Path, alias_path: Path) -> None:
    from graphrag.cache import AliasMemoryCache, L2SemanticCache, L3_ENTITY_CACHE

    l2 = L2SemanticCache(database_path=l2_path)
    alias = AliasMemoryCache(database_path=alias_path)
    print("当前缓存状态：")
    print(f"- L2 语义缓存：{l2.stats()}")
    print(f"- 别名记忆缓存：{alias.stats()}")
    print(f"- 当前进程 L3 实体缓存：{L3_ENTITY_CACHE.stats()}")
    print("- L1 短期问答缓存：后端进程内存缓存，需通过重启后端清空。")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理危险货物问答系统缓存。")
    parser.add_argument("--l2", action="store_true", help="清理 L2 语义问答缓存。")
    parser.add_argument("--alias", action="store_true", help="清理别名记忆缓存。")
    parser.add_argument(
        "--l3",
        action="store_true",
        help="清理当前脚本进程内的 L3 实体缓存，并提示重启后端。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="清理 L2、别名记忆缓存和当前脚本进程内的 L3。",
    )
    parser.add_argument("--stats", action="store_true", help="只查看缓存状态，不清理。")
    parser.add_argument("--l2-db", type=Path, default=default_l2_path(), help="L2 SQLite 文件路径。")
    parser.add_argument(
        "--alias-db",
        type=Path,
        default=default_alias_path(),
        help="别名记忆 SQLite 文件路径。",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.stats:
        show_stats(args.l2_db, args.alias_db)
        return 0

    clear_default = not any([args.l2, args.alias, args.l3, args.all])
    should_clear_l2 = args.all or args.l2 or clear_default
    should_clear_alias = args.all or args.alias or clear_default
    should_clear_l3 = args.all or args.l3

    if should_clear_l2:
        clear_l2(args.l2_db)
    if should_clear_alias:
        clear_alias(args.alias_db)
    if should_clear_l3:
        clear_l3_memory()

    print("")
    print("如果你想确保 L1/L3 内存缓存也完全消失，请重启后端：")
    print("./venv/bin/python ./src/run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
