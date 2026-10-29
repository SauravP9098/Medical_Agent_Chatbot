from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from app.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

@lru_cache()
def clients():
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise ValueError("Groq and Pinecone API Keys must be set in the environment varibales.")
    return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_chat_model), Pinecone(api_key=settings.pinecone_api_key)

@lru_cache()
def embedder():
    return HuggingFaceEmbeddings(model_name=settings.huggingface_embed_model)

def ensure_index():
    _, pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(name=settings.pinecone_index, dimension=384, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    return pc.Index(settings.pinecone_index)

def embed(texts):
    return embedder().embed_documents(texts)

def search_kb(query, top_k = 5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text', ''), 'score': round(m.score,3), 'source': 'Gala Encylopedia (2002)'} for m in result.matches if m.score >= 0.30]
