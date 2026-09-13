from pydantic import BaseModel, ConfigDict, Field

class ChunkBase(BaseModel):
    content: str
    source: str
    embedding: list[float]

class ChunkCreate(ChunkBase):
    pass

class ChunkRead(ChunkBase):
    model_config = ConfigDict(from_attributes=True)

    id: int

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)

class SearchHit(BaseModel):
    content: str
    source: str
    score: float
    # which retriever found it, and at what position. None = that side missed it
    # entirely; both-None is impossible. Kept in the response because "sparse is
    # null on every hit" is the symptom of the filter silently matching nothing.
    dense_rank: int | None
    sparse_rank: int | None
    # cross-encoder relevance. Raw logit, not a probability: sign is meaningful
    # (>0 leans relevant) but the scale is not comparable between queries, so order
    # by it and never threshold on it. A whole result set scoring negative means the
    # reranker found nothing it liked, which is worth surfacing to the caller.
    rerank_score: float
