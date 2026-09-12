from pydantic import BaseModel, ConfigDict, EmailStr

from models.models import Role

class RegisterRequest(BaseModel):
    gmail: EmailStr
    username: str
    password: str

class AdminCreateUser(RegisterRequest):
    role: Role = Role.user

class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    gmail: EmailStr
    username: str
    role: Role

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
