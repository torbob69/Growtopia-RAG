import dotenv
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

JWT_SECRET_KEY = dotenv.get_key('C:/Growtopia-RAG/fast-api-backend/vector-service/.env', 'JWT_SECRET_KEY')
JWT_ALGORITHM = "HS256"

# tokenUrl just tells Swagger's "Authorize" button where to fetch a token from;
# auth-service issues the tokens, vector-service only ever verifies them.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://127.0.0.1:8003/login")

async def require_admin(token: str = Depends(oauth2_scheme)) -> None:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("role") != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
