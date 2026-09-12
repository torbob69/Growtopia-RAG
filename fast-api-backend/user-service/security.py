import bcrypt
import dotenv
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from models.models import User

JWT_SECRET_KEY = dotenv.get_key('C:/Growtopia-RAG/fast-api-backend/user-service/.env', 'JWT_SECRET_KEY')
JWT_ALGORITHM = "HS256"

# tokenUrl just tells Swagger's "Authorize" button where to fetch a token from;
# auth-service issues the tokens, user-service only ever verifies them.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://127.0.0.1:8003/login")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_session),
) -> User:
    credentials_exception = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
    except jwt.InvalidTokenError:
        raise credentials_exception
    if user_id is None:
        raise credentials_exception

    user = await db.get(User, int(user_id))
    if user is None:
        raise credentials_exception
    return user
