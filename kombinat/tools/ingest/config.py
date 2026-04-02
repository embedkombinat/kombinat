from __future__ import annotations

from pydantic import BaseModel


class IngestConfig(BaseModel):
    # Source
    dataset_name: str = "nomic-ai/nomic-embed-unsupervised-data"
    split: str = "squad"
    max_docs: int | None = None

    # Retrieval
    bm25_top_k: int = 10_000
    dense_top_k: int = 10_000
    rrf_k: int = 60
    candidates_per_query: int = 5_000

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    embedding_device: str = "mps"

    # FAISS
    faiss_index_dir: str = "~/.kombinat/faiss_indexes"
    faiss_min_search_docs: int = 100_000

    # Writer
    db_batch_size: int = 5000

    # Database
    database_url: str = ""

    @property
    def source_dataset_label(self) -> str:
        """e.g. 'nomic-ai/nomic-embed-unsupervised-data/squad'"""
        return f"{self.dataset_name}/{self.split}"
