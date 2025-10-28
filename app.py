import streamlit as st
import os

from pinecone import Pinecone

# Importaciones de LlamaIndex Core
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer

# Importaciones de Conectores (RUTAS ESTABLES CON LA VERSIÓN FIJADA)
from llama_index.vector_stores.pinecone import PineconeVectorStore
# Importación del LLM (Ruta exacta del paquete instalado)
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Agente RAG Bancario (Gratuito)",
    page_icon="🏦",
    layout="centered"
)
st.title("🏦 Agente RAG Bancario")
st.caption("Asistente impulsado por la documentación interna del Banco Caroní.")

# --- 1. CONFIGURACIÓN DE CREDENCIALES (Streamlit Secrets) ---

# Se leen las claves del archivo .streamlit/secrets.toml
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "manuales-banco-rag")

# Modelos que usamos (deben coincidir con la ingesta)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/gemma-7b-it" # LLM rápido y robusto para el free tier

# --- 2. INICIALIZACIÓN DE SERVICIOS ---

@st.cache_resource
def initialize_services():
    # 1. Inicializa Pinecone y el Vector Store
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # 2. Configura el Embedder (384 dim)
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        device="cpu"
    )

    # 3. Configura el LLM
    llm = HuggingFaceInferenceAPI(
        model_name=LLM_MODEL_NAME,
        token=HUGGINGFACE_TOKEN,
    )

    # 4. Configura LlamaIndex (Settings define los modelos a usar globalmente)
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 5. Crea el Índice (estructura de consulta que usa el Vector Store)
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index

# Inicializa el índice y lo guarda en la caché de Streamlit
index = initialize_services()

# --- 3. INTERFAZ DE CHAT ---

# Crea el motor de chat/consulta solo si no existe en la sesión
if "chat_engine" not in st.session_state:
    # Usamos memoria para mantener el contexto de la conversación
    memory = ChatMemoryBuffer.from_defaults(token_limit=10000)

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", # Modo ideal para RAG conversacional
        memory=memory,
        system_prompt=(
            "Eres un agente de asistencia bancaria, amable y profesional. "
            "Tu única fuente de conocimiento son los manuales proporcionados del Banco Caroní. "
            "Responde de forma concisa y basada estrictamente en el contexto recuperado."
        ),
    )

# Inicializa la lista de mensajes si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra el historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("¿Qué deseas saber sobre los manuales del banco?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Lógica de respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            # Llama al motor de chat para obtener la respuesta RAG
            # CORRECCIÓN: Usar .chat() en lugar de .query()
            response = st.session_state.chat_engine.chat(prompt)
            st.markdown(response.response)

    st.session_state.messages.append({"role": "assistant", "content": response.response})