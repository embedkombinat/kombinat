from uuid import UUID

from pydantic import BaseModel, Field


class AnnotationIn(BaseModel):
    pair_id: UUID
    label: int = Field(ge=0, le=3)
    input_tokens: int
    output_tokens: int
    raw_response_hash: str


class AnnotationSubmission(BaseModel):
    batch_id: UUID
    model_id: str
    quantization: str
    annotations: list[AnnotationIn]


class AnnotationResult(BaseModel):
    accepted: int
    rejected: int
    honeypot_accuracy: float | None
    pairs_verified: int
    contributor_tokens: dict[str, int]
