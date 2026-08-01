import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from .settings import settings


class Config:
    """检索系统的集中配置。"""

    TEXT_DIR = "./data/text"
    TEXT_FILE = "./data/GB+6944.txt"  # 兼容旧路径；新索引优先读取 TEXT_DIR 下的所有 txt。
    TABLE_DIR = "./data/tables/GB6944"
    INDEX_SAVE_PATH = "./data/text/faiss_index.bin"
    METADATA_SAVE_PATH = "./data/text/metadata.json"

    EMBEDDING_MODEL = settings.model.embedding_model
    RERANK_MODEL = settings.model.rerank_model

    CHUNK_MAX_LENGTH = 450
    CHUNK_OVERLAP = 80
    RETRIEVE_TOP_K = 20
    FINAL_TOP_K = 3
    LOW_CONFIDENCE_SCORE = -3.0
    SOURCE_NAME = "GB 6944-2025"

    NEO4J_URI = settings.neo4j.uri
    NEO4J_USER = settings.neo4j.user
    NEO4J_PASSWORD = settings.neo4j.password


QueryRoute = Literal["kg", "direct", "hybrid"]
SourceFilter = Literal["gb6944", "appendix_a", "all"]
QueryTarget = Literal[
    "packing_group",
    "hazard_class",
    "subsidiary_hazard",
    "special_provisions",
    "limited_quantities",
    "excepted_quantities",
    "packing_instruction",
    "special_packing_provisions",
    "portable_tank",
    "name",
    "definition",
    "reason",
    "requirement",
    "general",
]

SUBJECT_TYPES = {"un_number", "entity_name"}
ROUTES = {"direct", "kg", "hybrid"}
QUERY_TARGETS = {
    "packing_group",
    "hazard_class",
    "subsidiary_hazard",
    "special_provisions",
    "limited_quantities",
    "excepted_quantities",
    "packing_instruction",
    "special_packing_provisions",
    "portable_tank",
    "name",
    "definition",
    "reason",
    "requirement",
    "general",
}

COMPREHENSIVE_TARGETS = {
    "packing_group",
    "hazard_class",
    "subsidiary_hazard",
    "special_provisions",
    "excepted_quantities",
}

DOMAIN_WORDS = [
    "危险货物", "危险类别", "危险主类", "项别", "分类码", "联合国编号", "UN编号",
    "包装组", "包装类别", "包装等级", "次要危险性", "有限数量", "例外数量",
    "特殊规定", "包装规范", "包装指令", "移动罐体", "易燃液体", "腐蚀性物质",
    "爆炸品", "气体", "毒性物质", "感染性物质", "放射性物质", "杂项危险物质",
    "自反应物质", "自燃物质", "遇水放出易燃气体", "氧化性物质", "有机过氧化物",
    "GB 6944", "GB 12268", "品名表", "分类和品名编号",
]

KG_ATTR_KEYWORDS = [
    "包装组", "包装类别", "包装等级", "危险类别", "危险主类", "危险货物类别", "项别", "分类码",
    "类别", "分类", "归类", "哪一类", "哪类", "第几类", "几类", "属于哪一类", "属于哪类",
    "次要危险性", "UN编号", "联合国编号", "中文名称", "英文名称", "正式名称",
    "有限数量", "例外数量", "特殊规定", "包装规范", "包装指令", "运输名称",
]

DEFINITION_KEYWORDS = ("什么是", "是什么", "定义", "含义", "是指", "概念", "何为")
DEFINITION_QUERY_STOPWORDS = set(DEFINITION_KEYWORDS) | {"什么"}
TEXT_DEFINITION_MARKERS = ("是指", "即", "指的是", "定义为", "含义是", "是指的")
EXPLANATION_KEYWORDS = (
    "为什么", "原因", "依据", "含义", "是什么意思", "什么意思",
    "解释", "说明", "怎么判断", "判定依据", "要求", "规定",
)
DIRECT_REFERENCE_KEYWORDS = ("特殊规定", "附录", "表", "条款", "第")
TOKEN_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "为", "在", "中",
    "请问", "请", "问",
}
QUERY_CORE_STOPWORDS = (
    "请问", "的定义", *DEFINITION_KEYWORDS,
    "属于哪一类", "属于哪类", "属于", "哪一类", "哪类", "第几类", "几类",
    "危险货物", "危险品", "货物", "？", "?", "。", "，", ",",
)
LEXICAL_QUERY_STOPWORDS = set(DEFINITION_QUERY_STOPWORDS) | {
    "请问", "属于", "哪一类", "哪类", "危险货物",
}
QUERY_EXPANSION_SYNONYMS: Dict[str, str] = {
    "包装类别": "包装组 包装等级",
    "包装组": "包装类别 包装等级",
    "UN号": "联合国编号 UN编号",
    "UN编号": "联合国编号 UN号",
    "危险类别": "危险主类 项别 分类码",
    "泄漏": "泄漏 应急处理 处置",
    "定义": "是指 含义 概念",
}
EXACT_EXPLANATION_CLAUSES: List[Dict[str, Any]] = [
    {
        "source_name": "GB6944",
        "keywords": ("包装类别",),
        "match": "any",
        "section_ids": ("4.1.2",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("第3类", "易燃液体"),
        "match": "any",
        "section_ids": ("5.3.1.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("第1类", "爆炸品"),
        "match": "any",
        "section_ids": ("5.1.1.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("第2类",),
        "match": "any",
        "section_ids": ("5.2.1.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("第7类", "放射性"),
        "match": "any",
        "section_ids": ("5.7",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("第8类", "腐蚀性"),
        "match": "any",
        "section_ids": ("5.8.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("第9类", "杂项危险"),
        "match": "any",
        "section_ids": ("5.9.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("4.1项", "易燃固体", "自反应物质", "固态退敏爆炸品", "聚合性物质"),
        "match": "any",
        "section_ids": ("5.4.1.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("4.2项", "易于自燃"),
        "match": "any",
        "section_ids": ("5.4.1.2",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("4.3项", "遇水放出易燃气体"),
        "match": "any",
        "section_ids": ("5.4.1.3",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("5.1项", "氧化性物质"),
        "match": "any",
        "section_ids": ("5.5.1.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("5.2项", "有机过氧化物"),
        "match": "any",
        "section_ids": ("5.5.1.2",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("6.1项", "毒性物质"),
        "match": "any",
        "section_ids": ("5.6.1.1",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("6.2项", "感染性物质"),
        "match": "any",
        "section_ids": ("5.6.1.2",),
        "score": 1000.0,
    },
    {
        "source_name": "GB6944",
        "keywords": ("危险货物类别 项别 分类",),
        "match": "any",
        "section_ids": ("4.1.1",),
        "score": 950.0,
    },
]
EXPLANATION_SORT_HINTS: List[Dict[str, Any]] = [
    {
        "source_name": "GB6944",
        "keywords": ("包装类别",),
        "boost_contains": (
            ("4.1.2", 50.0),
            ("4.1 危险货物类别、项别和包装类别", 20.0),
        ),
        "penalty_regex": (
            (r"5\.[2-9].*包装类别", -20.0),
        ),
    },
    {
        "source_name": "GB6944",
        "keywords": ("第3类", "易燃液体"),
        "boost_contains": (
            ("5.3 第3类易燃液体", 50.0),
        ),
        "penalty_regex": (
            (r"5\.[2456789] 第", -20.0),
        ),
    },
]

TARGET_KEYWORDS: List[Tuple[QueryTarget, List[str]]] = [
    ("packing_group", ["包装组", "包装类别", "包装等级"]),
    ("hazard_class", [
        "危险类别", "危险主类", "危险货物类别", "项别", "分类码", "类别", "分类", "归类", "哪一类", "哪类", "第几类", "几类",
        "属于哪一类", "属于哪类",
    ]),
    ("subsidiary_hazard", ["次要危险性", "次要危险"]),
    ("limited_quantities", ["有限数量", "限量"]),
    ("excepted_quantities", ["例外数量", "例外数量编码"]),
    ("special_packing_provisions", ["特殊包装规定"]),
    ("portable_tank", ["移动罐体特殊规定", "移动罐体", "罐体特殊规定", "罐体"]),
    ("special_provisions", ["特殊规定", "特殊规定编号"]),
    ("packing_instruction", ["包装规范", "包装指令"]),
    ("name", ["中文名称", "英文名称", "正式名称", "运输名称", "品名"]),
    ("reason", ["为什么", "原因", "依据", "怎么判断", "判定依据"]),
    ("definition", list(DEFINITION_KEYWORDS)),
    ("requirement", ["要求", "规定", "如何", "怎么", "怎样", "注意", "处理", "应急", "泄漏"]),
]

TARGET_EXPLANATION_HINTS: Dict[QueryTarget, str] = {
    "packing_group": "包装类别 包装组 危险性先后顺序 划分依据",
    "hazard_class": "危险货物类别 分类 项别 判定依据",
    "subsidiary_hazard": "次要危险性 危险性先后顺序",
    "special_provisions": "特殊规定 附录A 条文说明",
    "limited_quantities": "有限数量 包装 限量运输",
    "excepted_quantities": "例外数量 例外数量包装编码",
    "packing_instruction": "包装规范 包装指令 包装要求",
    "special_packing_provisions": "特殊包装规定 包装要求",
    "portable_tank": "移动罐体 罐体特殊规定",
    "name": "正式运输名称 中文名称 英文名称",
    "definition": "定义 含义 概念",
    "reason": "判定依据 原因 解释",
    "requirement": "要求 规定 注意事项",
    "general": "",
}

TARGET_CANONICAL_LABELS: Dict[QueryTarget, str] = {
    "packing_group": "包装类别",
    "hazard_class": "危险类别",
    "subsidiary_hazard": "次要危险性",
    "special_provisions": "特殊规定",
    "limited_quantities": "有限数量",
    "excepted_quantities": "例外数量编码",
    "packing_instruction": "包装规范",
    "special_packing_provisions": "特殊包装规定",
    "portable_tank": "移动罐体特殊规定",
    "name": "正式运输名称",
    "definition": "定义",
    "reason": "判定依据",
    "requirement": "要求",
    "general": "相关信息",
}

ENTITY_ALIASES: Dict[str, str] = {
    # 常见俗称/简称。这里先保守维护，避免别名过度扩张造成误匹配。
    "酒精": "乙醇溶液",
    "酒精溶液": "乙醇溶液",
    "油漆": "涂料",
    "油漆稀释剂": "涂料相关材料",
}

COMMON_DG_NAMES = [
    "乙醇溶液", "酒精溶液", "2-二甲氨基乙醇", "二甲氨基乙醇",
    "2-二乙氨基乙醇", "二乙氨基乙醇", "二丁氨基乙醇",
    "乙醇胺溶液", "乙醇胺", "2-氯乙醇", "硝化甘油乙醇溶液",
    "乙醇和汽油混合物", "汽油", "乙醇", "酒精", "硫酸", "盐酸", "硝酸",
    "柴油", "丙酮", "甲醇", "苯", "甲苯", "二甲苯", "氢氧化钠",
    "氢氧化钾", "液化石油气",
]

DOMAIN_ENTITIES = [
    "易燃液体", "腐蚀性物质", "爆炸品", "气体", "易燃固体", "毒性物质", "感染性物质", "放射性物质", "氧化性物质", "有机过氧化物", "危险货物",
]

CLASS_NAMES = {
    "1": "第1类 爆炸品",
    "2": "第2类 气体",
    "3": "第3类 易燃液体",
    "4": "第4类 易燃固体、易于自燃的物质、遇水放出易燃气体的物质",
    "5": "第5类 氧化性物质和有机过氧化物",
    "6": "第6类 毒性物质和感染性物质",
    "7": "第7类 放射性物质",
    "8": "第8类 腐蚀性物质",
    "9": "第9类 杂项危险物质和物品",
}

DIVISION_NAMES = {
    "2.1": "易燃气体",
    "2.2": "非易燃无毒气体",
    "2.3": "毒性气体",
    "4.1": "易燃固体、自反应物质、固态退敏爆炸品和聚合物质",
    "4.2": "易于自燃的物质",
    "4.3": "遇水放出易燃气体的物质",
    "5.1": "氧化性物质",
    "5.2": "有机过氧化物",
    "6.1": "毒性物质",
    "6.2": "感染性物质",
}

TEXT_ONLY_PREFIXES = ("GB", "GBT")
GB12268_TABLE_PATH = Path("data/tables/GB12268/GB+12268_tab.csv")
EXCEPTED_QUANTITY_TABLE_PATH = Path("data/tables/GB12268/GB+12268_tab1.csv")
SPECIAL_PROVISIONS_APPENDIX_PATH = Path("data/text/GB+12268附录A.txt")

UN_PATTERN = re.compile(r"(?i)(?:UN\s*[-－]?\s*)?(?<!\d)(\d{4})(?!\d)")
GB6944_REFERENCE_PATTERN = re.compile(r"(?i)(?<![A-Z0-9])(?:GB|GBT|GB/T)?\s*[-＋+]?\s*6944(?!\d)")
GB12268_REFERENCE_PATTERN = re.compile(r"(?i)(?<![A-Z0-9])(?:GB|GBT|GB/T)?\s*[-＋+]?\s*12268(?!\d)")

TARGET_FIELD_MAP: Dict[QueryTarget, Dict[str, Any]] = {
    # 用户说“类别”“项别”“危险类别”时，优先使用 GB 12268 原表中的
    # class_or_division 精确值；如 6.1 应回答“6.1 毒性物质”，而不是只回答“第6类”。
    # 解释来源到 GB 6944。
    "hazard_class": {
        "label": "类别或项别",
        "field": "class_or_division_labels",
        "source": "GB6944",
    },
    "subsidiary_hazard": {
        "label": "次要危险性",
        "field": "subsidiary_hazard",
        "source": "GB6944",
    },
    "packing_group": {
        "label": "包装类别",
        "field": "packing_groups",
        "source": "GB6944",
    },
    "special_provisions": {
        "label": "特殊规定",
        "field": "special_provisions",
        "source": "GB12268_APPENDIX",
    },
    "excepted_quantities": {
        "label": "例外数量",
        "field": "excepted_quantities",
        "source": "GB12268_EXCEPTED_QUANTITY_TABLE",
    },
}

KG_EVIDENCE_TITLE = "【证据A：结构化图谱事实】"
KG_KEY_FIELDS = [
    "联合国编号 (UN号)",
    "中文正式名称",
    "英文正式名称",
    "类别或项别",
    "所属危险主类",
    "所属危险主类别",
    "次要危险性",
    "允许使用的所有包装组别",
    "特殊规定条文索引",
    "例外数量编码",
    "有限数量限制",
    "包装规范指令",
    "特殊包装规定",
    "移动罐体特殊规定",
]

KG_BASE_FIELDS = [
    "联合国编号 (UN号)",
    "中文正式名称",
]

EXPLICIT_KG_TARGET_FIELDS: Dict[QueryTarget, List[str]] = {
    "packing_group": ["允许使用的所有包装组别"],
    "hazard_class": ["类别或项别", "所属危险主类", "所属危险主类别", "次要危险性"],
    "subsidiary_hazard": ["次要危险性", "类别或项别"],
    "special_provisions": ["特殊规定条文索引"],
    "limited_quantities": ["有限数量限制"],
    "excepted_quantities": ["例外数量编码"],
    "packing_instruction": ["包装规范指令"],
    "special_packing_provisions": ["特殊包装规定"],
    "portable_tank": ["移动罐体特殊规定"],
    "name": ["英文正式名称"],
}
