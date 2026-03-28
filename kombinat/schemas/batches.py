from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from kombinat.schemas.pairs import PairBrief


class BatchClaimRequest(BaseModel):
    size: int = Field(default=100, ge=1, le=500)


class BatchOut(BaseModel):
    batch_id: UUID
    expires_at: datetime
    pairs: list[PairBrief]
