import pickle
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import Chunk

async def seeding(db: AsyncSession):
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
