from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from models.models import EMBEDDING_DIM
from schemas.schemas import SearchHit, SearchRequest
from security import require_admin
from service import search, seeding

# Must stay the model the chunks were embedded with, or the query lands in a different
# space and every result is quietly wrong. EMBEDDING_DIM is the only automatic guard.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Reranks the retrieved shortlist. Unlike the embedder this one has no compatibility
# constraint with what is stored — it reads text, not vectors — so it can be swapped
# freely without re-seeding. Chosen over BAAI/bge-reranker-base: equal on a 7-query
# eval (7/7 each), roughly 3x faster.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ponytail: both on CPU. A query embed is ~30ms and a 20-chunk rerank ~630ms, which keeps
# the GPU free for the pipeline's bulk re-embeds. Reranking dominates /search latency —
# move it to cuda (~3x faster) before anything else if the endpoint gets hot.
_model: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    m = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    dim = m.get_sentence_embedding_dimension()
    if dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"{EMBEDDING_MODEL} emits {dim}-d vectors but chunk.embedding is {EMBEDDING_DIM}-d"
        )
    _model["embedder"] = m
    _model["reranker"] = CrossEncoder(RERANK_MODEL, device="cpu")
    yield
    _model.clear()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def health():
    return "vector service is ok"


@app.post('/seed', status_code=201, dependencies=[Depends(require_admin)])
async def seed(db: AsyncSession = Depends(get_session)):
    await seeding(db)


@app.post('/search', response_model=list[SearchHit])
async def search_chunks(req: SearchRequest, db: AsyncSession = Depends(get_session)):
    return await search(db, req.query, req.top_k, _model["embedder"], _model["reranker"])
