import pickle
from fastapi import HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import Chunk
from schemas.schemas import SearchHit

async def seeding(db: AsyncSession):
    # ponytail: "already seeded" is derived from the table itself instead of a
    # separate flag/table — one less thing to keep in sync. Doesn't protect
    # against two concurrent /seed calls racing past this check; add a
    # transaction-level lock if that becomes a real scenario.
    already_seeded = await db.execute(select(Chunk.id).limit(1))
    if already_seeded.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "vector store already seeded")

    with open('C:/Growtopia-RAG/pipeline/docs_splitted.pkl', 'rb') as f:
        docs_splitted = pickle.load(f)
    with open('C:/Growtopia-RAG/pipeline/vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
        
    chunks = [
        Chunk(content=doc.page_content, source=doc.metadata['source'], embedding=vector)
        for doc, vector in zip(docs_splitted, vectors)
    ]
    db.add_all(chunks)
    await db.commit()


CANDIDATES = 50   # per retriever, before fusion
RRF_K = 60        # standard damping constant; bigger = flatter, less top-heavy
DF_FRAC = 0.01    # a lexeme in >1% of chunks is not a proper noun; sparse ignores it.
                  # Relative, not an absolute count: every re-crawl grows the corpus and a
                  # fixed cutoff silently changes meaning as it does.

# Dense (pgvector cosine) + sparse (Postgres FTS) fused with Reciprocal Rank Fusion.
# RRF sums 1/(k+rank) per retriever, so it never compares a cosine distance against a
# ts_rank score, only positions — the two scales are unrelated and normalising them is
# guesswork.
#
# Sparse only ever sees the RARE lexemes of the query, ANDed. Postgres ts_rank_cd has no
# IDF and no tf saturation (unlike real BM25), so handing it a whole question lets a chunk
# repeating "fish" 30x outrank the one chunk actually titled "Mint". Restricting it to
# proper nouns plays to the one thing sparse beats dense at.
HYBRID = text("""
WITH q AS (
    SELECT CAST(:qvec AS vector) AS qv,
           -- CAST is load-bearing: asyncpg types the placeholder from the bigint that
           -- count(*) returns, so a bare :df_frac binds 0.01 as int 0, making df_max 0
           -- and silently abstaining sparse on every query. No error, just no sparse.
           (SELECT count(*) * CAST(:df_frac AS float) FROM chunk) AS df_max
),
rare AS (
    SELECT l FROM unnest(tsvector_to_array(to_tsvector('english', :query))) l
    WHERE (SELECT count(*) FROM chunk c WHERE c.fts @@ plainto_tsquery('english', l))
          <= (SELECT df_max FROM q)
),
tsq AS (
    -- NULL when the query holds no rare terms at all -> sparse abstains, dense decides
    SELECT CASE WHEN count(*) = 0 THEN NULL
                ELSE array_to_string(array_agg(l), ' & ')::tsquery END AS t FROM rare
),
dense AS (
    SELECT id, ROW_NUMBER() OVER () AS rank FROM (
        SELECT id FROM chunk, q ORDER BY embedding <=> q.qv LIMIT :n
    ) d
),
sparse AS (
    SELECT id, ROW_NUMBER() OVER () AS rank FROM (
        SELECT c.id FROM chunk c, tsq WHERE tsq.t IS NOT NULL AND c.fts @@ tsq.t
        ORDER BY ts_rank_cd(c.fts, tsq.t) DESC LIMIT :n
    ) s
)
SELECT c.content, c.source, d.rank AS dense_rank, s.rank AS sparse_rank,
       COALESCE(1.0 / (:rrf_k + d.rank), 0) + COALESCE(1.0 / (:rrf_k + s.rank), 0) AS score
FROM dense d
FULL OUTER JOIN sparse s ON d.id = s.id      -- full outer: keep hits either side found alone
JOIN chunk c ON c.id = COALESCE(d.id, s.id)
ORDER BY score DESC
LIMIT :top
""")


async def search(db: AsyncSession, query: str, top_k: int, embedder) -> list[SearchHit]:
    vector = embedder.encode(query, normalize_embeddings=True)
    rows = (await db.execute(HYBRID, {
        "qvec": "[" + ",".join(map(str, vector)) + "]",
        "query": query,
        "n": CANDIDATES,
        "rrf_k": RRF_K,
        "top": top_k,
        "df_frac": DF_FRAC,
    })).fetchall()
    return [SearchHit(content=r.content, source=r.source, score=r.score,
                      dense_rank=r.dense_rank, sparse_rank=r.sparse_rank) for r in rows]
