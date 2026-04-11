from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import numpy as np

    from kombinat.tools.ingest.source import Corpus


class BM25Index(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: BM25Okapi
    doc_ids: list[str]


def build_bm25_index(corpus: Corpus) -> BM25Index:
    """Tokenize documents and build BM25Okapi index."""
    tokenized = [doc.lower().split() for doc in corpus.doc_texts]
    return BM25Index(
        index=BM25Okapi(tokenized),
        doc_ids=corpus.doc_ids,
    )


def bm25_retrieve(index: BM25Index, query: str, top_k: int) -> list[tuple[str, float]]:
    """Return [(doc_id, score), ...] sorted by score descending."""
    tokenized_query = query.lower().split()
    scores: np.ndarray = index.index.get_scores(tokenized_query)
    n = min(top_k, len(index.doc_ids))
    top_indices = scores.argsort()[-n:][::-1]
    return [(index.doc_ids[i], float(scores[i])) for i in top_indices]
