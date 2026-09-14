from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import Base, engine, get_session
from schemas.schemas import ChatRequest, ChatResponse, InteractionRead, SessionRead
from security import current_user_id, oauth2_scheme
from service import answer, http, list_interactions, list_sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ponytail: create_all rather than the alembic scaffold auth- and vector-service carry.
    # Two tables, no migration history, nothing in production to migrate. Switch to alembic
    # the first time a column has to change on a database that already holds rows.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await http.aclose()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def health():
    return {"status": "chat service is ok"}


@app.post('/chat', response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_session),
    user_id: int = Depends(current_user_id),
    # the caller's raw token, forwarded to /search so vector-service authorises the user
    # itself. FastAPI resolves oauth2_scheme once and reuses it for current_user_id above.
    token: str = Depends(oauth2_scheme),
):
    return await answer(db, req.query, req.session_id, user_id, token)


@app.get('/sessions', response_model=list[SessionRead])
async def sessions(
    db: AsyncSession = Depends(get_session),
    user_id: int = Depends(current_user_id),
):
    return await list_sessions(db, user_id)


@app.get('/sessions/{session_id}/interactions', response_model=list[InteractionRead])
async def interactions(
    session_id: int,
    db: AsyncSession = Depends(get_session),
    user_id: int = Depends(current_user_id),
):
    return await list_interactions(db, session_id, user_id)
