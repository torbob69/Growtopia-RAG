from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import Theme, User
from security import hash_password, verify_password

async def update_username(db: AsyncSession, user: User, new_username: str) -> User:
    user.username = new_username
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "username already taken")
    await db.refresh(user)
    return user

async def update_password(db: AsyncSession, user: User, current_password: str, new_password: str) -> User:
    if not verify_password(current_password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "current password is incorrect")
    user.password_hash = hash_password(new_password)
    await db.commit()
    await db.refresh(user)
    return user

async def update_theme(db: AsyncSession, user_id: int, theme: Theme) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    user.theme = theme
    await db.commit()
    await db.refresh(user)
    return user
