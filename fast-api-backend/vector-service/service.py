import pickle
from sqlalchemy.ext.asyncio import AsyncSession

def seeding(db : AsyncSession):
    vectors = pickle.load('C:/Growtopia-RAG/pipeline/vectors.pkl')
    docs_splitted = pickle.load('C:/Growtopia-RAG/pipeline/docs_splitted.pkl')
    
    