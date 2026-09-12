from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from models.models import User
from schemas.schemas import PasswordUpdate, ThemeUpdate, UserRead, UsernameUpdate
from security import get_current_user
from service import update_password, update_theme, update_username

app = FastAPI()

@app.get('/health')
def health():
    return "user service is ok"

@app.patch('/me/username', response_model=UserRead)
async def change_username(
    payload: UsernameUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    return await update_username(db, user, payload.new_username)

@app.patch('/me/password', response_model=UserRead)
async def change_password(
    payload: PasswordUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    return await update_password(db, user, payload.current_password, payload.new_password)

# no auth: theme is a cosmetic preference, not worth gating behind a login
@app.patch('/users/{user_id}/theme', response_model=UserRead)
async def change_theme(
    user_id: int,
    payload: ThemeUpdate,
    db: AsyncSession = Depends(get_session),
):
    return await update_theme(db, user_id, payload.theme)
