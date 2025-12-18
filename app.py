import streamlit as st
import sys

# --- FIX DE EMERGENCIA V3: BYPASS LÓGICO ---
# En lugar de intentar borrar archivos (que falla por permisos),
# engañamos a Python para que crea que el plugin conflictivo no existe.
# Esto evita que Pinecone lance el error DeprecatedPluginError.
sys.modules["pinecone_plugins.inference"] = None
# -----------------------------------------------------------

import json
import time
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
from modules.admin import open_admin_modal
from modules.library import open_library_modal

# --- FUNCIÓN UI: Cierra modales al interactuar ---
def close_modals():
    st.session_state.is_library_open = False
    st.session_state.is_admin_open = False

# Configuración Página
st.set_page_config(page_title="Asistente Operacional Inteligente", page_icon="🏦", layout="centered")

# Login
authenticator, auth_status, username, credentials = run_login()

if auth_status is False: st.error('Credenciales incorrectas.')
elif auth_status is None: st.warning('Ingrese usuario y contraseña.')
elif auth_status is True:
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = credentials['usernames'][username]['id']
    
    user_role = credentials['usernames'][username]['role']
    
    # 1. CARGAMOS MODELOS BASE (LLM + Retriever)
    llm, retriever, vector_store = load_models_and_retriever()
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
        st.session_state.messages = []

    # ==========================================
    # BARRA LATERAL
    # ==========================================
    
    st.sidebar.markdown(f"### 👤 {credentials['usernames'][username]['name']}")
    authenticator.logout('Cerrar Sesión', 'sidebar')
    st.sidebar.divider()

    # --- INICIALIZACIÓN DE ESTADOS ---
    if "is_library_open" not in st.session_state: st.session_state.is_library_open = False
    if "is_admin_open" not in st.session_state: st.session_state.is_admin_open = False

    # 1. GESTIÓN DE CONVERSACIONES
    st.sidebar.subheader("💬 Mis Conversaciones")
    
    if st.sidebar.button("Iniciar Nuevo Chat", icon=":material/add_circle:", type="primary", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        close_modals()
        st.rerun()

    sessions = get_user_sessions(username)
    with st.sidebar.container(height=300, border=True):
        if not sessions: st.caption("📭 No hay historial reciente.")
        for sess in sessions:
            is_active = st.session_state.current_session_id == sess['session_id']
            btn_type = "primary" if is_active else "secondary"
            icon_sess = ":material/folder_open:" if is_active else ":material/chat_bubble_outline:"
            
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                if st.button(sess['title'], key=f"btn_{sess['session_id']}", icon=icon_sess, type=btn_type, use_container_width=True):
                    st.session_state.current_session_id = sess['session_id']
                    st.session_state.messages = load_chat_history(sess['session_id'])
                    close_modals() 
                    st.rerun()
            with c2:
                if st.button("", key=f"del_{sess['session_id']}", icon=":material/delete:", help="Borrar conversación"):
                    delete_session(sess['session_id'])
                    if is_active:
                        st.session_state.current_session_id = None; st.session_state.messages = []
                    close_modals()
                    st.rerun()

    st.sidebar.write("") 
    st.sidebar.divider()

    # 2. BIBLIOTECA
    if st.sidebar.button("Biblioteca de Documentos", icon=":material/library_books:", use_container_width=True):
        st.session_state.is_library_open = True
        st.session_state.is_admin_open = False 
        st.rerun()

    # 3. NOTAS RÁPIDAS
    st.sidebar.subheader("🛠️ Herramientas")
    with st.sidebar.expander("📝 Crear Nota Rápida"):
        
        if st.session_state.get("note_upload_success", False):
            st.success("Nota guardada correctamente.", icon=":material/check_circle:")
            st.session_state.note_upload_success = False 

        with st.form("quick_note_form", clear_on_submit=True):
            nt = st.text_input("Título Nota", placeholder="Ej: Solución Error 503")
            nc = st.text_area("Contenido", placeholder="Describe la solución paso a paso...")
            
            submitted = st.form_submit_button("Guardar Nota", icon=":material/save:", use_container_width=True)
            
            if submitted:
                if nt and nc:
                    try:
                        docs, ids = process_text_to_docs(nc, nt)
                        vector_store.add_documents(docs, ids=ids)
                        
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        
                        safe_title = nt.encode('latin-1', 'replace').decode('latin-1')
                        safe_content = nc.encode('latin-1', 'replace').decode('latin-1')
                        
                        pdf.set_font("Arial", "B", 16)
                        pdf.multi_cell(0, 10, safe_title)
                        pdf.ln(5)
                        
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, safe_content)
                        
                        path = f"{username}/notas_{sanitize_filename_to_ascii(nt)}.pdf"
                        supabase_admin.storage.from_("manuales-pdf").upload(
                            path, 
                            bytes(pdf.output()), 
                            {"upsert":"true", "content-type": "application/pdf"}
                        )
                        
                        supabase_admin.table('manuales').upsert({
                            'filename': f"{nt}.pdf", 'storage_path': path, 'uploader_username': username, 
                            'vector_count': len(docs), 'file_size': len(bytes(pdf.output()))
                        }, on_conflict='storage_path').execute()
                        
                        st.session_state.note_upload_success = True
                        close_modals()
                        st.rerun()
                        
                    except Exception as e: 
                        st.error(f"Error al guardar nota: {e}", icon=":material/error:")
                else:
                    st.warning("Debes llenar título y contenido.", icon=":material/warning:")

    # 4. ADMIN PANEL
    if user_role == 'admin':
        st.sidebar.write("")
        if st.sidebar.button("Panel de Administrador", icon=":material/admin_panel_settings:", type="primary", use_container_width=True):
            st.session_state.is_admin_open = True
            st.session_state.is_library_open = False 
            st.rerun()

    # ==========================================
    # ÁREA PRINCIPAL DE CHAT
    # ==========================================
    st.title("🏦 Asistente Operacional Inteligente")
    # Texto descriptivo combinado
    st.markdown("""
    Sistema que analiza la biblioteca de manuales y normativas del banco para que te olvides de buscar en múltiples PDFs. 
    Simplemente consulta sobre **configuraciones, fallas o procedimientos**, y recibirás una solución sintetizada al instante basada en la documentación oficial vigente.
    """)

    # --- 1. HISTORIAL DE MENSAJES ---
    if st.session_state.current_session_id is None:
        st.info("👋 **Bienvenido.** Selecciona una conversación del historial o inicia un **Nuevo Chat** para comenzar.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    st.write("") 
    st.markdown("---")

    # --- 2. CONTROLES DE CHAT ---
    c_mode, c_chat_type = st.columns(2)
    
    with c_mode:
        system_options = ["🧠 Híbrido (IA + Manuales)", "📜 Estricto (Solo Manuales)"]
        selected_system_mode = st.radio(
            "Modo de Respuesta:", 
            system_options, 
            horizontal=True, 
            label_visibility="collapsed", 
            key="sys_mode",
            on_change=close_modals
        )
        
    with c_chat_type:
        chat_options = ["⚡ Puntual (Gasto: 1x)", "💬 Conversacional (Gasto: 2x)"]
        selected_chat_mode = st.radio(
            "Tipo de Chat:", 
            chat_options, 
            horizontal=True, 
            label_visibility="collapsed", 
            key="chat_mode",
            on_change=close_modals
        )

    # --- 3. GENERACIÓN DE LA CADENA ---
    rag_chain = create_rag_chain(llm, retriever, selected_system_mode, selected_chat_mode)

    # --- 4. INPUT DEL CHAT ---
    prompt = st.chat_input("Escribe tu consulta...")

    if prompt:
        close_modals()

    # --- RENDERIZADO DE MODALES ---
    if st.session_state.is_library_open:
        open_library_modal(username, vector_store)
        
    if st.session_state.is_admin_open:
        open_admin_modal(username, credentials)

    # --- PROCESAMIENTO DEL MENSAJE ---
    if prompt:
        if st.session_state.current_session_id is None:
            nid = create_new_session(username, prompt)
            if nid: st.session_state.current_session_id = nid
            else: st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        save_message(username, st.session_state.current_session_id, "user", prompt)

        with st.chat_message("assistant"):
            sys_name = selected_system_mode.split(' ')[1] 
            chat_name = "Puntual" if "Puntual" in selected_chat_mode else "Conversacional"
            
            try:
                with st.spinner(f"Analizando ({sys_name} | {chat_name})..."):
                    hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
                    
                    resp = rag_chain.invoke({"input": prompt, "chat_history": hist})
                    
                    ans = resp["answer"]
                    st.markdown(ans)
                    ctx = resp.get("context", [])
                    with st.expander("🔍 Fuentes consultadas", expanded=False):
                        if ctx:
                            for i, d in enumerate(ctx):
                                st.caption(f"**Fuente {i+1}:** {d.metadata.get('source')} | {d.page_content[:200]}...")
                        else: st.caption("No se encontraron fuentes en los manuales (o respuesta general).")
                    
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    save_message(username, st.session_state.current_session_id, "assistant", ans)
            
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str:
                    st.warning("⏳ **Límite de Capa Gratuita Alcanzado (Gemini Flash)**")
                    st.markdown("Has alcanzado el límite de consultas diarias o por minuto. Espera 60s.")
                else:
                    st.error(f"❌ Ocurrió un error inesperado: {error_str}", icon=":material/error:")