from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kombinat.tools.ingest.source import Corpus

if TYPE_CHECKING:
    import pathlib

import hashlib


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


DOCS = [
    "Paris is the capital of France and its largest city.",
    "Berlin is the capital of Germany, located on the River Spree.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
]

QUERIES = [
    "What is the capital of France?",
    "Tell me about the Eiffel Tower",
    "What programming language is used for ML?",
]

_DOC_IDS = [_sha(d) for d in DOCS]

_POSITIVE_DOC_IDS = [
    _sha(DOCS[0]),  # Paris doc
    _sha(DOCS[2]),  # Eiffel doc
    _sha(DOCS[4]),  # Python doc
]


@pytest.fixture
def tiny_corpus() -> Corpus:
    """5 documents, 3 queries, known positives."""
    doc_id_to_idx = {did: i for i, did in enumerate(_DOC_IDS)}
    return Corpus(
        doc_ids=list(_DOC_IDS),
        doc_texts=list(DOCS),
        queries=list(QUERIES),
        positive_doc_ids=list(_POSITIVE_DOC_IDS),
        split="squad",
        doc_id_to_idx=doc_id_to_idx,
    )


@pytest.fixture
def tmp_faiss_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Temporary directory for FAISS index files."""
    d = tmp_path / "faiss_indexes"
    d.mkdir()
    return d
