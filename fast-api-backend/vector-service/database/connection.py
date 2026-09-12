import dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

DB_PASSWORD = dotenv.get_key('C:/Growtopia-RAG/database-vector-rag/.env', 'DB_PASSWORD')
DB_URL = f"postgresql+psycopg://admin:{DB_PASSWORD}@localhost:5433/rag-chunk-embedding"

engine = create_async_engine(DB_URL)

class Base(DeclarativeBase):
    pass

def get_session():
    with AsyncSession(engine) as session:
        yield session