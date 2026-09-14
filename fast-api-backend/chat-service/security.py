import dotenv
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

JWT_SECRET_KEY = dotenv.get_key('C:/Growtopia-RAG/.env', 'JWT_SECRET_KEY')
JWT_ALGORITHM = "HS256"

# tokenUrl just tells Swagger's "Authorize" button where to fetch a token from;
# auth-service issues the tokens, chat-service only ever verifies them.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://127.0.0.1:8003/login")

async def require_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Any authenticated caller. Stateless: this service has no access to the auth DB,
    so a token stays valid here until it expires even if the user is deleted there."""
    try:
        # PyJWT verifies `exp` itself and raises InvalidTokenError once past it
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("sub") is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def current_user_id(payload: dict = Depends(require_user)) -> int:
    """The `sub` claim as an int. auth-service sets it to str(user.id); anything else
    reaching this point means a token this service should not be trusting."""
    try:
        return int(payload["sub"])
    except (TypeError, ValueError):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
