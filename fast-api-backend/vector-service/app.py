from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from service import seeding

app = FastAPI()

@app.get('/health')
def health():
    return "vector service is ok"

@app.post('/seed')
def seed(db : AsyncSession):
    return seeding(db)