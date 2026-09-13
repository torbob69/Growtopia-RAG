from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from models.models import EMBEDDING_DIM
from schemas.schemas import SearchHit, SearchRequest
from security import require_admin
from service import search, seeding

# Must stay the model the chunks were embedded with, or the query lands in a different
# space and every result is quietly wrong. EMBEDDING_DIM is the only automatic guard.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ponytail: CPU. One short query is ~30ms, and it keeps the GPU free for the
# pipeline's bulk re-embeds. Switch to cuda if /search ever gets hot.
_model: dict[str, SentenceTransformer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    m = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    dim = m.get_sentence_embedding_dimension()
    if dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"{EMBEDDING_MODEL} emits {dim}-d vectors but chunk.embedding is {EMBEDDING_DIM}-d"
        )
    _model["embedder"] = m
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
    return await search(db, req.query, req.top_k, _model["embedder"])
