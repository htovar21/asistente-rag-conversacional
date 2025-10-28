import streamlit as st
import os

from pinecone import Pinecone

# Importaciones de LlamaIndex Core
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer

# Importaciones de Conectores
from llama_index.vector_stores.pinecone import PineconeVectorStore
# CAMBIO: Importar el LLM de Gemini
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
# (Sin cambios aquí)
st.set_page_config(...)
st.title(...)
st.caption(...)

# --- 1. CONFIGURACIÓN DE CREDENCIALES (Streamlit Secrets) ---

# CAMBIO: Añadir la clave de Google AI
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "manuales-banco-rag")

# Modelos que usamos
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# CAMBIO: Usar un modelo Gemini disponible (Flash es rápido y gratuito)
LLM_MODEL_NAME = "models/gemini-1.5-flash-latest" # O 'models/gemini-pro'

# --- 2. INICIALIZACIÓN DE SERVICIOS ---

@st.cache_resource
def initialize_services():
    # 1. Inicializa Pinecone (Sin cambios)
    pc = Pinecone(...)
    pinecone_index = pc.Index(...)
    vector_store = PineconeVectorStore(...)

    # 2. Configura el Embedder (Sin cambios)
    embed_model = HuggingFaceEmbedding(...)

    # 3. CAMBIO: Configura el LLM de Gemini
    # LlamaIndex usa la variable de entorno GOOGLE_API_KEY por defecto,
    # pero podemos pasarla explícitamente si es necesario.
    # Asegúrate de que GOOGLE_API_KEY esté en tus secretos.
    # os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Alternativa si LlamaIndex no la toma
    llm = Gemini(model_name=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)


    # 4. Configura LlamaIndex (Sin cambios)
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 5. Crea el Índice (Sin cambios)
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index

# (El resto del código de la interfaz de chat sigue igual)
# ...
index = initialize_services()
# ... (Interfaz de Chat) ...