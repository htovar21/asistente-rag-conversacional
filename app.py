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
st.set_page_config(page_title="Agente RAG Bancario", page_icon="🏦", layout="centered")

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

    # ==========================
    # BARRA LATERAL (LIMPIA)
    # ==========================
    st.sidebar.title(f'Hola, {credentials["usernames"][username]["name"]}')
    authenticator.logout('Salir', 'sidebar')
    st.sidebar.divider()

    # 1. BOTÓN BIBLIOTECA (MODAL) - ACCESIBLE A TODOS
    if st.sidebar.button("📂 Biblioteca de Documentos", use_container_width=True):
        open_library_modal(username, vector_store)

    st.sidebar.divider()

    # 2. HISTORIAL DE CHATS
    st.sidebar.subheader("💬 Mis Chats")
    if st.sidebar.button("➕ Nuevo Chat", type="secondary", use_container_width=True):
        st.session_state.current_session_id = None; st.session_state.messages = []; st.rerun()

    sessions = get_user_sessions(username)
    with st.sidebar.container(height=200):
        if not sessions: st.caption("Sin historial.")
        for sess in sessions:
            c1, c2 = st.columns([0.8, 0.2])
            isActive = st.session_state.current_session_id == sess['session_id']
            if c1.button(f"📝 {sess['title']}", key=f"b_{sess['session_id']}", disabled=isActive):
                st.session_state.current_session_id = sess['session_id']
                st.session_state.messages = load_chat_history(sess['session_id'])
                st.rerun()
            if c2.button("✖️", key=f"d_{sess['session_id']}"):
                delete_session(sess['session_id']); st.rerun()

    st.sidebar.divider()

    # 3. NOTAS RÁPIDAS (Se mantiene en sidebar como pediste)
    with st.sidebar.expander("📝 Nota Rápida"):
        nt = st.text_input("Título"); nc = st.text_area("Contenido")
        if st.button("Guardar Nota"):
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
                    st.success("Nota guardada.")
                except Exception as e: st.error(f"Error: {e}")

    # 4. SUBIR MANUALES (Se mantiene en sidebar como pediste)
    with st.sidebar.expander("⬆️ Subir Manuales"):
        render_upload_section(username, vector_store, key_suffix="sidebar")

    # 5. ADMIN (Botón Modal)
    if user_role == 'admin':
        render_admin_panel(username, credentials)

    # ==========================
    # CHAT PRINCIPAL
    # ==========================
    st.title("🏦 Asistente Operacional")
    
    if st.session_state.current_session_id is None:
        st.info("👋 Inicia un nuevo chat para comenzar.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Consulta..."):
        if st.session_state.current_session_id is None:
            nid = create_new_session(username, prompt)
            if nid: st.session_state.current_session_id = nid
            else: st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        save_message(username, st.session_state.current_session_id, "user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
                resp = rag_chain.invoke({"input": prompt, "chat_history": hist})
                
                ans = resp["answer"]
                st.markdown(ans)
                
                ctx = resp.get("context", [])
                with st.expander("🔍 Fuentes"):
                    for i, d in enumerate(ctx):
                        st.caption(f"**Fuente {i+1}:** {d.metadata.get('source')} | {d.page_content[:200]}...")

                st.session_state.messages.append({"role": "assistant", "content": ans})
                save_message(username, st.session_state.current_session_id, "assistant", ans)