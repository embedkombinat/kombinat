from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel

if TYPE_CHECKING:
    from kombinat.tools.ingest.config import IngestConfig
    from kombinat.tools.ingest.fusion import RankedCandidate
    from kombinat.tools.ingest.source import Corpus


class CandidatePair(BaseModel):
    pair_id: str
    query_text: str
    doc_id: str
    doc_text: str
    source_dataset: str
    retrieval_method: str
    source_rank: int


def build_candidates(
    query: str,
    positive_doc_id: str,
    rrf_results: list[RankedCandidate],
    corpus: Corpus,
    config: IngestConfig,
) -> list[CandidatePair]:
    """Filter out the known positive, take top-N, build CandidatePair objects."""
    source_dataset = config.source_dataset_label
    candidates: list[CandidatePair] = []
    for rank, rc in enumerate(rrf_results):
        if rc.doc_id == positive_doc_id:
            continue
        if len(candidates) >= config.candidates_per_query:
            break
        doc_idx = corpus.doc_id_to_idx[rc.doc_id]
        pair_id = str(uuid5(NAMESPACE_URL, f"{query}|{rc.doc_id}|{source_dataset}"))
        candidates.append(
            CandidatePair(
                pair_id=pair_id,
                query_text=query,
                doc_id=rc.doc_id,
                doc_text=corpus.doc_texts[doc_idx],
                source_dataset=source_dataset,
                retrieval_method="bm25+dense",
                source_rank=rank + 1,
            )
        )
    return candidates
