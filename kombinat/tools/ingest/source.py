from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from datasets import load_dataset  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kombinat.tools.ingest.config import IngestConfig


class Corpus(BaseModel):
    """Deduplicated document corpus from one split."""

    doc_ids: list[str]
    doc_texts: list[str]
    queries: list[str]
    positive_doc_ids: list[str]
    split: str
    doc_id_to_idx: dict[str, int] = Field(default_factory=dict)


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_split(config: IngestConfig) -> Corpus:
    """Load a split from HuggingFace, deduplicate documents, return Corpus.

    Uses streaming=True so only the requested split is downloaded, row by row.
    Stops downloading immediately when max_docs unique documents are collected.
    """
    dataset = load_dataset(config.dataset_name, split=config.split, streaming=True)

    doc_id_to_text: dict[str, str] = {}
    doc_id_order: list[str] = []

    queries: list[str] = []
    positive_doc_ids: list[str] = []

    for row in dataset:
        doc_text: str = row["document"]
        query: str = row["query"]
        did = _doc_id(doc_text)

        if did not in doc_id_to_text:
            doc_id_to_text[did] = doc_text
            doc_id_order.append(did)

        queries.append(query)
        positive_doc_ids.append(did)

        if config.max_docs is not None and len(queries) >= config.max_docs:
            break

    doc_ids = doc_id_order
    doc_texts = [doc_id_to_text[did] for did in doc_ids]
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    return Corpus(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        queries=queries,
        positive_doc_ids=positive_doc_ids,
        split=config.split,
        doc_id_to_idx=doc_id_to_idx,
    )
