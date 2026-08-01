from dataclasses import dataclass

import torch

from .settings import apply_runtime_environment

apply_runtime_environment()

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RetrievalModels:
    """集中持有嵌入模型与重排模型，隔离模型加载细节。"""

    embedding: SentenceTransformer
    reranker_tokenizer: object
    reranker: object


def load_models(embedding_path: str, reranker_path: str) -> RetrievalModels:
    print(f"正在加载 BGE-M3 嵌入模型: {embedding_path}")
    embedding = SentenceTransformer(embedding_path)

    print(f"正在加载交叉验证重排模型: {reranker_path}")
    tokenizer = AutoTokenizer.from_pretrained(reranker_path)
    reranker = AutoModelForSequenceClassification.from_pretrained(reranker_path)
    if torch.cuda.is_available():
        reranker = reranker.cuda()
    reranker.eval()

    return RetrievalModels(
        embedding=embedding,
        reranker_tokenizer=tokenizer,
        reranker=reranker,
    )
