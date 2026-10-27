import os
import certifi
from dotenv import load_dotenv

load_dotenv()

ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_api_key: str = ''
    pincone_api_key: str = ''
    pincone_index : str = 'medical-agent-chatbot'
    ollama_chat_model : str = 'gpt-oss:120b'
    huggingface_embed_model : str = 'sentence-transformers/all-MiniLM-L6-v2'
    ollama_vision_model : str = 'gemma4:31b'
    api_token : str = ''
    allowed_origins : str = 'http://localhost:8000,http://127.0.0.1:8000'
    model_config= SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()
