from uuid import UUID

from pydantic import BaseModel


class PairBrief(BaseModel):
    pair_id: UUID
    query_text: str
    doc_text: str
    source_dataset: str
    is_honeypot: bool = False
