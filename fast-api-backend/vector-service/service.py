import pickle
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import Chunk

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
