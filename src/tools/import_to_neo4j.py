import os
import re
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphrag.definitions import CLASS_NAMES, DIVISION_NAMES
from graphrag.settings import settings


# ==================== 1. 路径与数据库配置 ====================
CSV_PATH = "./data/tables/GB12268/GB+12268_tab.csv"
NEO4J_URI = settings.neo4j.uri
NEO4J_USER = settings.neo4j.user
NEO4J_PASSWORD = settings.neo4j.password

REGULATION = {
    "standard_no": "GB 12268-2025",
    "name": "危险货物品名表",
    "year": "2025",
}

INVALID_VALUES = {"", "-", "—", "－", "/", "nan", "none", "null", "无"}


def clean_value(value) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return "" if text.lower() in INVALID_VALUES else text


def unique_values(values):
    seen = set()
    result = []
    for value in values:
        cleaned = clean_value(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def normalize_un_number(value) -> str:
    raw = clean_value(value).upper().replace(" ", "")
    if not raw:
        return ""
    raw = raw[2:] if raw.startswith("UN") else raw
    raw = raw.split(".")[0]
    if not raw.isdigit():
        return ""
    return f"UN{raw.zfill(4)}"


def normalize_packing_group(value) -> str:
    raw = clean_value(value)
    mapping = {
        "1": "Ⅰ", "I": "Ⅰ", "Ⅰ": "Ⅰ", "一": "Ⅰ",
        "2": "Ⅱ", "II": "Ⅱ", "Ⅱ": "Ⅱ", "二": "Ⅱ",
        "3": "Ⅲ", "III": "Ⅲ", "Ⅲ": "Ⅲ", "三": "Ⅲ",
    }
    return mapping.get(raw.upper(), mapping.get(raw, raw))


def normalize_class_or_division(value) -> str:
    return clean_value(value).upper()


def derive_class_id(class_or_division: str) -> str:
    match = re.match(r"^(\d+)", class_or_division)
    return match.group(1) if match else ""


def requirement_id_for(row: dict) -> str:
    parts = [
        row["un_number"],
        row["class_or_division"] or "NO_DIVISION",
        row["packing_group"] or "NO_PACKING_GROUP",
        row["limited_quantities"] or "NO_LIMITED_QUANTITY",
        row["packing_instruction"] or "NO_PACKING_INSTRUCTION",
    ]
    joined = "|".join(parts)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fffⅠⅡⅢ|_.-]+", "_", joined)


class HazardGraphImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def initialize_database(self):
        """初始化数据库：清空旧数据并建立唯一性约束和索引。"""
        with self.driver.session() as session:
            print("正在清空数据库老旧节点...")
            session.run("MATCH (n) DETACH DELETE n")

            print("🔑 正在构建唯一性约束和检索索引...")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:DangerousGood) REQUIRE d.un_number IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:HazardClass) REQUIRE h.class_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:HazardDivision) REQUIRE v.division_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:PackingGroup) REQUIRE p.group_rating IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.standard_no IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:TransportRequirement) REQUIRE t.requirement_id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:DangerousGood) ON (d.name_zh)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:DangerousGood) ON (d.name_en)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (v:HazardDivision) ON (v.division_name)")

    def import_csv(self, csv_path):
        """读取表1 CSV 并注入图谱，保留同一 UN 编号下的多包装组/运输要求。"""
        if not os.path.exists(csv_path):
            print(f"❌ 错误：未在路径 {csv_path} 找到 CSV 文件！请检查路径。")
            return

        print(f"📖 正在读取数据文件: {csv_path} ...")
        df = pd.read_csv(csv_path).fillna("")

        cleaned_rows = []
        for _, row in df.iterrows():
            un_number = normalize_un_number(row["un_number"])
            if not un_number:
                continue

            class_or_division = normalize_class_or_division(row["class_or_division"])
            class_id = derive_class_id(class_or_division)
            packing_group = normalize_packing_group(row["packing_group"])

            cleaned_rows.append({
                "un_number": un_number,
                "name_zh": clean_value(row["name_zh"]),
                "name_en": clean_value(row["name_en"]),
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"第{class_id}类") if class_id else "",
                "class_or_division": class_or_division,
                "division_name": DIVISION_NAMES.get(class_or_division, class_or_division),
                "subsidiary_hazard": clean_value(row["subsidiary_hazard"]),
                "packing_group": packing_group,
                "special_provisions": clean_value(row["special_provisions"]),
                "limited_quantities": clean_value(row["limited_quantities"]),
                "excepted_quantities": clean_value(row["excepted_quantities"]),
                "packing_instruction": clean_value(row["packing_instruction"]),
                "special_packing_provisions": clean_value(row["special_packing_provisions"]),
                "portable_tank_instruction": clean_value(row["portable_tank_instruction"]),
                "portable_tank_special_provisions": clean_value(row["portable_tank_special_provisions"]),
            })

        goods = {}
        for row in cleaned_rows:
            goods.setdefault(row["un_number"], []).append(row)

        regulation_query = """
        MERGE (r:Regulation {standard_no: $standard_no})
        SET r.name = $name,
            r.year = $year,
            r.source_type = "national_standard"
        """

        dangerous_good_query = """
        MERGE (d:DangerousGood {un_number: $un_number})
        SET d.name_zh = $name_zh,
            d.name_en = $name_en,
            d.name_zh_aliases = $name_zh_aliases,
            d.name_en_aliases = $name_en_aliases,
            d.class_or_divisions = $class_or_divisions,
            d.packing_groups = $packing_groups,
            d.subsidiary_hazards = $subsidiary_hazards,
            d.special_provisions = $special_provisions,
            d.limited_quantities = $limited_quantities,
            d.excepted_quantities = $excepted_quantities,
            d.packing_instruction = $packing_instruction,
            d.special_packing_provisions = $special_packing_provisions,
            d.portable_tank_instruction = $portable_tank_instruction,
            d.portable_tank_special_provisions = $portable_tank_special_provisions,
            d.source = $source
        WITH d
        MATCH (r:Regulation {standard_no: $source})
        MERGE (d)-[:DEFINED_IN]->(r)
        """

        row_query = """
        MATCH (d:DangerousGood {un_number: $un_number})
        MATCH (r:Regulation {standard_no: $source})

        FOREACH (_ IN CASE WHEN $class_id <> "" THEN [1] ELSE [] END |
            MERGE (h:HazardClass {class_id: $class_id})
            SET h.class_name = $class_name,
                h.source = $source
            MERGE (d)-[b:BELONGS_TO]->(h)
            SET b.class_or_divisions = $class_or_divisions_for_class,
                b.subsidiary_hazards = $subsidiary_hazards_for_class,
                b.subsidiary_hazard = $subsidiary_hazards_for_class_text,
                b.source = $source
            MERGE (h)-[:DEFINED_IN]->(r)
        )

        FOREACH (_ IN CASE WHEN $class_or_division <> "" THEN [1] ELSE [] END |
            MERGE (v:HazardDivision {division_id: $class_or_division})
            SET v.division_name = $division_name,
                v.source = $source
            MERGE (d)-[:HAS_DIVISION]->(v)
            MERGE (v)-[:DEFINED_IN]->(r)
        )

        WITH d, r
        OPTIONAL MATCH (h:HazardClass {class_id: $class_id})
        OPTIONAL MATCH (v:HazardDivision {division_id: $class_or_division})
        FOREACH (_ IN CASE WHEN h IS NOT NULL AND v IS NOT NULL THEN [1] ELSE [] END |
            MERGE (v)-[:PART_OF]->(h)
        )

        WITH d, r
        FOREACH (_ IN CASE WHEN $packing_group <> "" THEN [1] ELSE [] END |
            MERGE (p:PackingGroup {group_rating: $packing_group})
            SET p.source = $source
            MERGE (d)-[:REQUIRES_PACKING]->(p)
            MERGE (p)-[:DEFINED_IN]->(r)
        )

        WITH d, r
        MERGE (t:TransportRequirement {requirement_id: $requirement_id})
        SET t.un_number = $un_number,
            t.class_or_division = $class_or_division,
            t.subsidiary_hazard = $subsidiary_hazard,
            t.packing_group = $packing_group,
            t.special_provisions = $special_provisions,
            t.limited_quantities = $limited_quantities,
            t.excepted_quantities = $excepted_quantities,
            t.packing_instruction = $packing_instruction,
            t.special_packing_provisions = $special_packing_provisions,
            t.portable_tank_instruction = $portable_tank_instruction,
            t.portable_tank_special_provisions = $portable_tank_special_provisions,
            t.source = $source
        MERGE (d)-[:HAS_TRANSPORT_REQUIREMENT]->(t)
        MERGE (t)-[:DEFINED_IN]->(r)
        WITH t
        OPTIONAL MATCH (v:HazardDivision {division_id: $class_or_division})
        OPTIONAL MATCH (p:PackingGroup {group_rating: $packing_group})
        FOREACH (_ IN CASE WHEN v IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t)-[:FOR_DIVISION]->(v)
        )
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t)-[:USES_PACKING_GROUP]->(p)
        )
        """

        print("📥 开始写入 Neo4j 数据库...")
        success_count = 0

        with self.driver.session() as session:
            session.run(regulation_query, **REGULATION)

            for un_number, rows in goods.items():
                names_zh = unique_values(row["name_zh"] for row in rows)
                names_en = unique_values(row["name_en"] for row in rows)
                divisions = unique_values(row["class_or_division"] for row in rows)
                packing_groups = unique_values(row["packing_group"] for row in rows)

                session.run(
                    dangerous_good_query,
                    un_number=un_number,
                    name_zh=names_zh[0] if names_zh else "",
                    name_en=names_en[0] if names_en else "",
                    name_zh_aliases=names_zh,
                    name_en_aliases=names_en,
                    class_or_divisions=divisions,
                    packing_groups=packing_groups,
                    subsidiary_hazards=unique_values(row["subsidiary_hazard"] for row in rows),
                    special_provisions=" / ".join(unique_values(row["special_provisions"] for row in rows)),
                    limited_quantities=" / ".join(unique_values(row["limited_quantities"] for row in rows)),
                    excepted_quantities=" / ".join(unique_values(row["excepted_quantities"] for row in rows)),
                    packing_instruction=" / ".join(unique_values(row["packing_instruction"] for row in rows)),
                    special_packing_provisions=" / ".join(unique_values(row["special_packing_provisions"] for row in rows)),
                    portable_tank_instruction=" / ".join(unique_values(row["portable_tank_instruction"] for row in rows)),
                    portable_tank_special_provisions=" / ".join(unique_values(row["portable_tank_special_provisions"] for row in rows)),
                    source=REGULATION["standard_no"],
                )

                rows_by_class = {}
                for row in rows:
                    rows_by_class.setdefault(row["class_id"], []).append(row)

                for row in rows:
                    class_rows = rows_by_class.get(row["class_id"], [])
                    session.run(
                        row_query,
                        **row,
                        source=REGULATION["standard_no"],
                        requirement_id=requirement_id_for(row),
                        class_or_divisions_for_class=unique_values(r["class_or_division"] for r in class_rows),
                        subsidiary_hazards_for_class=unique_values(r["subsidiary_hazard"] for r in class_rows),
                        subsidiary_hazards_for_class_text=" / ".join(unique_values(r["subsidiary_hazard"] for r in class_rows)),
                    )
                    success_count += 1

        print(f"📊 数据清洗流共处理了 {success_count} 行数据记录。")
        print(f"📌 共生成 {len(goods)} 个 DangerousGood 节点，并保留每行 TransportRequirement 明细。")
        print("=" * 50)


# ==================== 3. 执行入口 ====================
if __name__ == "__main__":
    importer = HazardGraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        importer.initialize_database()
        importer.import_csv(CSV_PATH)
    finally:
        importer.close()
