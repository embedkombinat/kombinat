from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Source
    dataset_name: str = "nomic-ai/nomic-embed-unsupervised-data"
    split: str = "squad"
    max_docs: int | None = None

    # Retrieval.
    # candidates_per_query is the project's annotation budget knob: every
    # candidate needs required_annotations (2) labels, so N candidates/query
    # multiplies the whole dataset by 2N. Contrastive training consumes a
    # handful of hard negatives per query (ANCE/CDE use single digits), and
    # deep-tail candidates are easy negatives that in-batch sampling already
    # covers for free — verifying them wastes contributor GPU time. Depth
    # beyond the fused top-N comes from re-mining with the improved model
    # (see the manifesto's iterative cycle), not from deeper static lists.
    bm25_top_k: int = 1_000
    dense_top_k: int = 1_000
    rrf_k: int = 60
    candidates_per_query: int = 10

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
