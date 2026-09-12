import enum

from sqlalchemy import Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base

class Role(str, enum.Enum):
    user = "user"
    admin = "admin"

class Theme(str, enum.Enum):
    light = "light"
    dark = "dark"

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[Role] = mapped_column(Enum(Role, name="role"), nullable=False)
    theme: Mapped[Theme] = mapped_column(Enum(Theme, name="theme"), nullable=False)
