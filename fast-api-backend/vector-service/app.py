from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from security import require_admin
from service import seeding

app = FastAPI()

@app.get('/health')
def health():
    return "vector service is ok"

@app.post('/seed', status_code=201, dependencies=[Depends(require_admin)])
async def seed(db: AsyncSession = Depends(get_session)):
    await seeding(db)