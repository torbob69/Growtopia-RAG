import dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

DB_PASSWORD = dotenv.get_key('C:/Growtopia-RAG/.env', 'DB_PASSWORD')
DB_URL = f"postgresql+asyncpg://admin:{DB_PASSWORD}@localhost:5435/chat-db"

engine = create_async_engine(DB_URL)

class Base(DeclarativeBase):
    pass

async def get_session():
    async with AsyncSession(engine) as session:
        yield session
