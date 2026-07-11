from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from kombinat.schemas.pairs import PairBrief


class BatchClaimRequest(BaseModel):
    size: int = Field(default=100, ge=1, le=500)
    # Model the annotator will label with (judge-diversity steering): when
    # provided, the claim prefers pairs that have no annotation from this
    # model's family yet. Optional — old clients that omit it get the
    # unsteered ordering.
    model_id: str | None = None


class BatchOut(BaseModel):
    batch_id: UUID
    expires_at: datetime
    pairs: list[PairBrief]
