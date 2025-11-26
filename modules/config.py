import os
import streamlit as st
from dotenv import load_dotenv

def load_config():
    """Carga y valida las variables de entorno/secretos."""
    try:
        # Intento carga local
        load_dotenv()
        config = {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag"),
            "SUPABASE_URL": os.getenv("SUPABASE_URL"),
            "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
            "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
            "COOKIE_SECRET_KEY": os.getenv("COOKIE_SECRET_KEY", "default_secret_key_123")
        }
        # Si falta alguna clave crítica, intentar cargar desde st.secrets (Streamlit Cloud)
        if not config["GOOGLE_API_KEY"]:
            raise ImportError
            
    except ImportError:
        # Carga desde Streamlit Cloud
        config = {
            "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
            "PINECONE_API_KEY": st.secrets["PINECONE_API_KEY"],
            "PINECONE_INDEX_NAME": st.secrets.get("PINECONE_INDEX_NAME", "manuales-banco-rag"),
            "SUPABASE_URL": st.secrets["SUPABASE_URL"],
            "SUPABASE_KEY": st.secrets["SUPABASE_KEY"],
            "SUPABASE_SERVICE_KEY": st.secrets["SUPABASE_SERVICE_KEY"],
            "COOKIE_SECRET_KEY": st.secrets.get("COOKIE_SECRET_KEY", "default_secret_key_123")
        }

    # Validación final
    required_keys = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_KEY"]
    if not all(config.get(k) for k in required_keys):
        st.error(f"Error: Faltan claves de configuración críticas: {required_keys}")
        st.stop()
        
    return config

# Instancia global de configuración
CONFIG = load_config()