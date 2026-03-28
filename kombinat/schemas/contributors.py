from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ContributorOut(BaseModel):
    id: UUID
    github_username: str
    github_avatar_url: str | None
    reputation_score: float
    total_annotations: int
    total_input_tokens: int
    total_output_tokens: int
    created_at: datetime
    last_seen_at: datetime


class AuthRequest(BaseModel):
    code: str
    state: str


class AuthResponse(BaseModel):
    access_token: str
    expires_in: int
    contributor: ContributorOut
