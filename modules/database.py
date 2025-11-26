import streamlit as st
from pinecone import Pinecone
from supabase import create_client, Client
from modules.config import CONFIG

def init_connections():
    try:
        # Cliente ANÓNIMO (Lectura pública/autenticación)
        supabase_anon: Client = create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_KEY"])
        
        # Cliente ADMIN (Escritura/Gestión total - USAR CON CUIDADO)
        supabase_admin: Client = create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_SERVICE_KEY"])
        
        # Conectar a Pinecone
        pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"])
        pinecone_index = pc.Index(CONFIG["PINECONE_INDEX_NAME"])
        
        return supabase_anon, supabase_admin, pinecone_index, pc
        
    except Exception as e:
        st.error(f"Error fatal al conectar con los servicios: {e}")
        st.stop()

# Inicializamos conexiones
supabase_anon, supabase_admin, pinecone_index, pinecone_client = init_connections()