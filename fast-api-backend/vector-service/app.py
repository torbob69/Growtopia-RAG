import os
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from models.models import EMBEDDING_DIM
from schemas.schemas import SearchHit, SearchRequest
from security import require_admin, require_user
from service import search, seeding

# Must stay the model the chunks were embedded with, or the query lands in a different
# space and every result is quietly wrong. EMBEDDING_DIM is the only automatic guard.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Reranks the retrieved shortlist. Unlike the embedder this one has no compatibility
# constraint with what is stored — it reads text, not vectors — so it can be swapped
# freely without re-seeding. Chosen over BAAI/bge-reranker-base: equal on a 7-query
# eval (7/7 each), roughly 3x faster.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reranking dominates /search latency, so it gets the GPU; the query embed is ~30ms either
# way and stays on CPU, leaving VRAM to the pipeline's bulk re-embeds. Falling back to CPU
# rather than failing: a machine without CUDA (or with a CPU-only torch wheel) should still
# serve, just slower. VECTOR_DEVICE overrides to pin it either way.
RERANK_DEVICE = os.getenv("VECTOR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

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
    ce = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)
    # CUDA builds kernels on the first forward pass, not at load: without this the first
    # real /search paid ~21s while the rest ran in ~230ms. Spend it at startup instead.
    ce.predict([("warmup", "warmup")])
    _model["reranker"] = ce
    print(f"reranker on {RERANK_DEVICE}", flush=True)
    yield
    _model.clear()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def health():
    # reports the device because "is the reranker actually on the GPU" is otherwise only
    # answerable by timing a request — it is a ~3x latency difference, silent either way
    return {"status": "vector service is ok", "rerank_device": RERANK_DEVICE}


@app.post('/seed', status_code=201, dependencies=[Depends(require_admin)])
async def seed(db: AsyncSession = Depends(get_session)):
    await seeding(db)


@app.post('/search', response_model=list[SearchHit], dependencies=[Depends(require_user)])
async def search_chunks(req: SearchRequest, db: AsyncSession = Depends(get_session)):
    return await search(db, req.query, req.top_k, _model["embedder"], _model["reranker"])
