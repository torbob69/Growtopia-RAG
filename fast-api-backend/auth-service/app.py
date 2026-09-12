from datetime import timedelta

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from models.models import Role
from schemas.schemas import AdminCreateUser, RegisterRequest, Token, UserRead
from security import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, get_current_user, require_role
from service import authenticate_user, register_user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
def health():
    return "auth service is ok"

@app.post('/register', response_model=UserRead, status_code=201)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_session)):
    # public signup always lands as "user" — admins are created via /admin/users
    return await register_user(db, payload.gmail, payload.username, payload.password, role=Role.user)

@app.post('/login', response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_session)):
    user = await authenticate_user(db, form_data.username, form_data.password)
    token = create_access_token(
        {"sub": str(user.id), "username": user.username, "role": user.role.value},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=token)

@app.get('/me', response_model=UserRead)
async def me(user=Depends(get_current_user)):
    return user

@app.post('/admin/users', response_model=UserRead, status_code=201)
async def admin_create_user(
    payload: AdminCreateUser,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(require_role(Role.admin)),
):
    return await register_user(db, payload.gmail, payload.username, payload.password, role=payload.role)
