from pydantic import BaseModel, ConfigDict

from models.models import Theme

class UsernameUpdate(BaseModel):
    new_username: str

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str

class ThemeUpdate(BaseModel):
    theme: Theme

class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    theme: Theme
