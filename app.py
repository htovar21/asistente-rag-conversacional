import streamlit as st
import json
from fpdf import FPDF
from langchain_core.messages import HumanMessage, AIMessage

# --- MÓDULOS ---
from modules.config import CONFIG
from modules.database import supabase_admin
from modules.auth import run_login
from modules.rag_engine import load_models_and_retriever, create_rag_chain
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii
from modules.chat_service import get_user_sessions, create_new_session, load_chat_history, save_message, delete_session

# NUEVOS MÓDULOS
from modules.admin import render_admin_panel
from modules.library import open_library_modal
from modules.uploader import render_upload_section

# Configuración Página
st.set_page_config(page_title="Asistente Operacional", page_icon="🏦", layout="centered")

# Login
authenticator, auth_status, username, credentials = run_login()

if auth_status is False: st.error('Credenciales incorrectas.')
elif auth_status is None: st.warning('Ingrese usuario y contraseña.')
elif auth_status is True:
    
    # --- INICIO USUARIO LOGUEADO ---
    user_role = credentials['usernames'][username]['role']
    llm, retriever, vector_store = load_models_and_retriever()
    rag_chain = create_rag_chain(llm, retriever)

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
        st.session_state.messages = []

    # ==========================================
    # BARRA LATERAL (REDISEÑADA)
    # ==========================================
    
    # 1. PERFIL DE USUARIO
    st.sidebar.markdown(f"### 👤 {credentials['usernames'][username]['name']}")
    authenticator.logout('Cerrar Sesión', 'sidebar')
    st.sidebar.divider()

    # 2. HISTORIAL DE CHATS (VISUALIZACIÓN PRO)
    st.sidebar.subheader("💬 Mis Conversaciones")
    
    # Botón principal destacado
    if st.sidebar.button("➕ Iniciar Nuevo Chat", type="primary", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()

    sessions = get_user_sessions(username)
    
    # Contenedor con borde para agrupar visualmente el historial
    with st.sidebar.container(height=300, border=True):
        if not sessions:
            st.caption("📭 No tienes historial reciente.")
        
        for sess in sessions:
            # Determinamos si este es el chat activo para pintarlo diferente
            is_active = st.session_state.current_session_id == sess['session_id']
            btn_type = "primary" if is_active else "secondary"
            icon = "📂" if is_active else "💭"
            
            # Layout de columnas: Botón Título (Grande) | Botón Borrar (Pequeño)
            col_chat, col_del = st.columns([0.85, 0.15])
            
            with col_chat:
                # El botón ocupa todo el ancho de su columna
                if st.button(f"{icon} {sess['title']}", key=f"btn_{sess['session_id']}", type=btn_type, use_container_width=True):
                    st.session_state.current_session_id = sess['session_id']
                    st.session_state.messages = load_chat_history(sess['session_id'])
                    st.rerun()
            
            with col_del:
                if st.button("🗑️", key=f"del_{sess['session_id']}", help="Eliminar chat"):
                    delete_session(sess['session_id'])
                    # Si borramos el activo, limpiamos la pantalla
                    if is_active:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()

    # 3. BIBLIOTECA (AHORA DEBAJO DE CHATS)
    st.sidebar.write("") # Espaciador
    if st.sidebar.button("📚 Abrir Biblioteca de Documentos", use_container_width=True):
        open_library_modal(username, vector_store)

    st.sidebar.divider()

    # 4. HERRAMIENTAS (AGRUPADAS)
    st.sidebar.subheader("🛠️ Herramientas")
    
    # Subir Manuales
    with st.sidebar.expander("⬆️ Cargar Conocimiento (PDF)"):
        render_upload_section(username, vector_store, key_suffix="sidebar")

    # Notas Rápidas
    with st.sidebar.expander("📝 Crear Nota Rápida"):
        nt = st.text_input("Título Nota")
        nc = st.text_area("Contenido")
        if st.button("Guardar Nota", use_container_width=True):
            if nt and nc:
                try:
                    docs, ids = process_text_to_docs(nc, nt)
                    vector_store.add_documents(docs, ids=ids)
                    
                    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 16); pdf.multi_cell(0, 10, nt)
                    pdf.set_font("Arial", size=12); pdf.multi_cell(0, 10, nc)
                    
                    path = f"{username}/notas_{sanitize_filename_to_ascii(nt)}.pdf"
                    supabase_admin.storage.from_("manuales-pdf").upload(path, bytes(pdf.output()), {"upsert":"true"})
                    
                    supabase_admin.table('manuales').upsert({
                        'filename': f"{nt}.pdf", 'storage_path': path, 'uploader_username': username, 
                        'vector_count': len(docs), 'file_size': len(bytes(pdf.output()))
                    }, on_conflict='storage_path').execute()
                    st.toast("✅ Nota guardada y procesada.")
                except Exception as e: st.error(f"Error: {e}")

    # 5. ADMIN (SOLO SI ES ROL ADMIN)
    if user_role == 'admin':
        render_admin_panel(username, credentials)

    # ==========================================
    # ÁREA PRINCIPAL DE CHAT
    # ==========================================
    st.title("🏦 Asistente Operacional")
    st.caption("Sistema de Apoyo basado en Manuales Internos")
    
    if st.session_state.current_session_id is None:
        # Pantalla de bienvenida vacía
        st.info("👋 ¡Hola! Selecciona un chat del historial o inicia uno nuevo para comenzar.")
    
    # Mostrar mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Input Usuario
    if prompt := st.chat_input("Escribe tu consulta..."):
        
        # Crear sesión si no existe
        if st.session_state.current_session_id is None:
            nid = create_new_session(username, prompt)
            if nid: st.session_state.current_session_id = nid
            else: st.stop()

        # Guardar y mostrar user msg
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        save_message(username, st.session_state.current_session_id, "user", prompt)

        # Respuesta IA
        with st.chat_message("assistant"):
            with st.spinner("Analizando manuales..."):
                hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
                resp = rag_chain.invoke({"input": prompt, "chat_history": hist})
                
                ans = resp["answer"]
                st.markdown(ans)
                
                ctx = resp.get("context", [])
                with st.expander("🔍 Fuentes consultadas", expanded=False):
                    if ctx:
                        for i, d in enumerate(ctx):
                            st.caption(f"**Fuente {i+1}:** {d.metadata.get('source')} | {d.page_content[:200]}...")
                    else:
                        st.caption("Respuesta generada con conocimiento general (sin manuales).")

                st.session_state.messages.append({"role": "assistant", "content": ans})
                save_message(username, st.session_state.current_session_id, "assistant", ans)