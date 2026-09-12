import dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# user-service reads/writes the users table auth-service owns and migrates —
# it never runs migrations of its own here.
DB_PASSWORD = dotenv.get_key('C:/Growtopia-RAG/database-auth/.env', 'DB_PASSWORD')
DB_URL = f"postgresql+asyncpg://admin:{DB_PASSWORD}@localhost:5434/auth-db"

engine = create_async_engine(DB_URL)

class Base(DeclarativeBase):
    pass

async def get_session():
    async with AsyncSession(engine) as session:
        yield session
