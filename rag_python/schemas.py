# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

from typing import Optional
from pydantic import BaseModel

class UploadResponse(BaseModel):
    status: str
    doc_id: str
    version: Optional[int] = None
    chunks: Optional[int] = None
    duplicate_chunks_skipped: Optional[int] = None
    near_dup_of: Optional[str] = None


class ChatRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    source: str
    # score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class SeedResponse(BaseModel):
    status: str
    passages_seen: int
    breakdown: dict[str, int]