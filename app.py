import streamlit as st
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

# Configuración Página
st.set_page_config(page_title="Asistente Operacional", page_icon="🏦", layout="centered")

# Login
authenticator, auth_status, username, credentials = run_login()

if auth_status is False: st.error('Credenciales incorrectas.')
elif auth_status is None: st.warning('Ingrese usuario y contraseña.')
elif auth_status is True:
    
    # --- VERSIÓN OPTIMIZADA (Gracias al cambio en auth.py) ---
    # Ya no hace falta consultar a Supabase de nuevo, el ID ya vino en el login
    if "user_id" not in st.session_state:
        st.session_state.user_id = credentials['usernames'][username]['id']
    
    # Resto del código...
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
        # FIX: Cerrar modales al iniciar nuevo chat
        st.session_state.is_library_open = False
        st.session_state.is_admin_open = False
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
                    # FIX: Cerrar modales al cambiar de chat
                    st.session_state.is_library_open = False 
                    st.session_state.is_admin_open = False
                    st.rerun()
            with c2:
                if st.button("", key=f"del_{sess['session_id']}", icon=":material/delete:", help="Borrar conversación"):
                    delete_session(sess['session_id'])
                    if is_active:
                        st.session_state.current_session_id = None; st.session_state.messages = []
                    
                    # --- FIX CRÍTICO: CERRAR MODALES AL BORRAR ---
                    st.session_state.is_library_open = False
                    st.session_state.is_admin_open = False
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
        
        # --- NUEVO: SISTEMA DE NOTIFICACIÓN PERSISTENTE ---
        # Si la carga fue exitosa en el ciclo anterior, mostramos el aviso ahora.
        if st.session_state.get("note_upload_success", False):
            st.success("Nota guardada correctamente.", icon=":material/check_circle:")
            st.session_state.note_upload_success = False # Apagar aviso para la próxima

        # USAMOS st.form PARA EVITAR RECARGAS MIENTRAS SE ESCRIBE
        with st.form("quick_note_form", clear_on_submit=True):
            nt = st.text_input("Título Nota", placeholder="Ej: Solución Error 503")
            nc = st.text_area("Contenido", placeholder="Describe la solución paso a paso...")
            
            # Botón de envío dentro del form
            submitted = st.form_submit_button("Guardar Nota", icon=":material/save:", use_container_width=True)
            
            if submitted:
                if nt and nc:
                    try:
                        # 1. Guardar en Vector Store (Memoria IA)
                        docs, ids = process_text_to_docs(nc, nt)
                        vector_store.add_documents(docs, ids=ids)
                        
                        # 2. Generar PDF (Con limpieza de caracteres para evitar error FPDF)
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        
                        # Limpieza: Reemplazamos caracteres no compatibles con Latin-1 (como emojis) por '?'
                        safe_title = nt.encode('latin-1', 'replace').decode('latin-1')
                        safe_content = nc.encode('latin-1', 'replace').decode('latin-1')
                        
                        pdf.set_font("Arial", "B", 16)
                        pdf.multi_cell(0, 10, safe_title)
                        pdf.ln(5)
                        
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, safe_content)
                        
                        # 3. Subir a Storage
                        path = f"{username}/notas_{sanitize_filename_to_ascii(nt)}.pdf"
                        supabase_admin.storage.from_("manuales-pdf").upload(
                            path, 
                            bytes(pdf.output()), 
                            {"upsert":"true", "content-type": "application/pdf"}
                        )
                        
                        # 4. Registrar en SQL (CORREGIDO PARA USAR ID)
                        supabase_admin.table('manuales').upsert({
                            'filename': f"{nt}.pdf", 
                            'storage_path': path, 
                            'uploader_id': st.session_state.user_id, # <--- CAMBIO AQUÍ TAMBIÉN
                            'vector_count': len(docs), 
                            'file_size': len(bytes(pdf.output()))
                        }, on_conflict='storage_path').execute()
                        
                        # GUARDAR ESTADO DE ÉXITO PARA MOSTRARLO TRAS RECARGA
                        st.session_state.note_upload_success = True
                        
                        # --- FIX CRÍTICO: CERRAR MODALES Y RECARGAR ---
                        st.session_state.is_library_open = False
                        st.session_state.is_admin_open = False
                        st.rerun()
                        
                    except Exception as e: 
                        # El error se muestra directo porque aquí no hacemos rerun
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
    st.title("🏦 Asistente Operacional")
    st.caption("Sistema Inteligente de Apoyo basado en Normativas")

    # --- 1. HISTORIAL DE MENSAJES (MOVIDO ARRIBA) ---
    if st.session_state.current_session_id is None:
        st.info("👋 **Bienvenido.** Selecciona una conversación del historial o inicia un **Nuevo Chat** para comenzar.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Espaciador visual
    st.write("") 
    st.markdown("---")

    # --- 2. CONTROLES DE CHAT (CONFIGURACIÓN) ---
    c_mode, c_chat_type = st.columns(2)
    
    with c_mode:
        # Modo de Sistema (Híbrido vs Estricto)
        system_options = ["🧠 Híbrido (IA + Manuales)", "📜 Estricto (Solo Manuales)"]
        selected_system_mode = st.radio("Modo de Respuesta:", system_options, horizontal=True, label_visibility="collapsed", key="sys_mode")
        
    with c_chat_type:
        # Modo de Chat (Conversacional vs Puntual - Ahorro)
        chat_options = ["⚡ Puntual (Gasto: 1x)", "💬 Conversacional (Gasto: 2x)"]
        selected_chat_mode = st.radio("Tipo de Chat:", chat_options, horizontal=True, label_visibility="collapsed", key="chat_mode")

    # --- 3. GENERACIÓN DE LA CADENA ---
    rag_chain = create_rag_chain(llm, retriever, selected_system_mode, selected_chat_mode)

    # --- 4. INPUT DEL CHAT (Pinned at Bottom) ---
    prompt = st.chat_input("Escribe tu consulta...")

    # Si hay input, cerramos inmediatamente cualquier modal
    if prompt:
        st.session_state.is_library_open = False
        st.session_state.is_admin_open = False

    # --- RENDERIZADO DE MODALES ---
    # Solo se muestran si la bandera es True.
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
            
            # --- TRY/EXCEPT MEJORADO CON LÍMITES REALES ---
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
                # Si es error 429 (Too Many Requests / ResourceExhausted)
                if "429" in error_str or "ResourceExhausted" in error_str:
                    st.warning("⏳ **Límite de Capa Gratuita Alcanzado (Gemini Flash)**")
                    st.markdown(
                        """
                        Has alcanzado uno de los límites del plan gratuito de Google Gemini:
                        
                        * **Velocidad:** Máx. 5 solicitudes/minuto.
                        * **Tokens:** Máx. 250k Tokens de entrada/minuto (TPM).
                        * **Diario:** Máx. 20 Solicitudes/día (RPD).
                        
                        👉 **Recomendación:** Espera **1 minuto** e intenta de nuevo usando el modo **⚡ Puntual**. 
                        
                        *Si el error persiste tras esperar, es probable que hayas agotado el cupo de 20 consultas del día.*
                        """
                    )
                else:
                    # Otros errores
                    st.error(f"❌ Ocurrió un error inesperado: {error_str}")