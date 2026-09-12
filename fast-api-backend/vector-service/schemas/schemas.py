from pydantic import BaseModel, ConfigDict

class ChunkBase(BaseModel):
    content: str
    source: str
    embedding: list[float]

class ChunkCreate(ChunkBase):
    pass

class ChunkRead(ChunkBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
