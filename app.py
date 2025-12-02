import streamlit as st
import json
import io
from pypdf import PdfReader
from fpdf import FPDF
from langchain_core.messages import HumanMessage, AIMessage

# --- IMPORTACIONES DE MÓDULOS PROPIOS ---
from modules.config import CONFIG
from modules.database import supabase_admin
from modules.auth import run_login
from modules.rag_engine import load_models_and_retriever, create_rag_chain
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii
from modules.admin import render_admin_panel
# Nueva importación para manejar sesiones
from modules.chat_service import get_user_sessions, create_new_session, load_chat_history, save_message, delete_session

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Agente RAG Bancario", page_icon="🏦", layout="centered")

# 1. Autenticación
authenticator, auth_status, username, credentials = run_login()

if auth_status is False:
    st.error('Usuario o contraseña incorrecto.')
elif auth_status is None:
    st.warning('Ingrese sus credenciales.')
    
elif auth_status is True:
    # ------------------
    # USUARIO LOGUEADO
    # ------------------
    user_role = credentials['usernames'][username]['role']
    
    # Cargar modelos IA
    llm, retriever, vector_store = load_models_and_retriever()
    rag_chain = create_rag_chain(llm, retriever)

    # --- GESTIÓN DE ESTADO DE SESIÓN (Inicialización) ---
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
        st.session_state.messages = []

    # ==========================================
    # BARRA LATERAL (SIDEBAR)
    # ==========================================
    st.sidebar.title(f'Hola, {credentials["usernames"][username]["name"]}')
    authenticator.logout('Cerrar Sesión', 'sidebar')
    
    # --- SECCIÓN 1: GESTIÓN DE CHATS (NUEVO) ---
    st.sidebar.divider()
    st.sidebar.subheader("💬 Historial de Chats")
    
    # Botón para crear nuevo chat (Limpia el ID actual y el historial en memoria)
    if st.sidebar.button("➕ Nuevo Chat", type="primary", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()

    # Listar sesiones anteriores desde la BD
    sessions = get_user_sessions(username)
    
    with st.sidebar.container(height=250): # Contenedor con scroll para los chats
        if not sessions:
            st.caption("No tienes chats previos.")
        for sess in sessions:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                # Estilo visual: si es el chat activo, desactivamos el botón
                is_active = st.session_state.current_session_id == sess['session_id']
                label = f"📝 {sess['title']}"
                if st.button(label, key=f"btn_{sess['session_id']}", disabled=is_active, use_container_width=True):
                    st.session_state.current_session_id = sess['session_id']
                    st.session_state.messages = load_chat_history(sess['session_id'])
                    st.rerun()
            with col2:
                if st.button("✖️", key=f"del_{sess['session_id']}", help="Borrar chat"):
                    delete_session(sess['session_id'])
                    # Si borramos el chat que estamos viendo, reseteamos la vista
                    if st.session_state.current_session_id == sess['session_id']:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()

    st.sidebar.divider()

    # --- SECCIÓN 2: AÑADIR NOTA RÁPIDA ---
    with st.sidebar.expander("📝 Añadir Nota Rápida"):
        note_title = st.text_input("Título Nota")
        note_text = st.text_area("Contenido Nota")
        
        if st.button("Subir Nota"):
            if note_title and note_text:
                with st.spinner("Procesando nota..."):
                    try:
                        # 1. Pinecone
                        docs, ids = process_text_to_docs(note_text, note_title)
                        vector_store.add_documents(docs, ids=ids)
                        
                        # 2. Generar PDF
                        pdf = FPDF(); pdf.add_page()
                        pdf.set_font("Helvetica", "B", 16)
                        pdf.multi_cell(0, 10, note_title.encode('latin-1', 'replace').decode('latin-1'))
                        pdf.ln(5)
                        pdf.set_font("Helvetica", size=12)
                        pdf.multi_cell(0, 10, note_text.encode('latin-1', 'replace').decode('latin-1'))
                        pdf_bytes = bytes(pdf.output())
                        
                        filename_ascii = sanitize_filename_to_ascii(note_title)
                        pdf_filename = f"{filename_ascii}.pdf"
                        storage_path = f"{username}/notas_{pdf_filename}"
                        
                        # 3. Supabase Storage
                        supabase_admin.storage.from_("manuales-pdf").upload(
                            path=storage_path, file=pdf_bytes, 
                            file_options={"upsert": "true", "content-type": "application/pdf"}
                        )
                        
                        # 4. Supabase SQL
                        supabase_admin.table('manuales').upsert({
                            'filename': pdf_filename, 'storage_path': storage_path, 
                            'uploader_username': username, 'vector_count': len(docs)
                        }, on_conflict='storage_path').execute()

                        st.success("Nota guardada exitosamente.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- SECCIÓN 3: SUBIR MANUAL PDF (MEJORADA CON FORMULARIO) ---
    with st.sidebar.expander("📂 Subir Manuales PDF"):
        # Usamos un formulario para agrupar la selección y el botón de acción
        with st.form("upload_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Seleccionar PDFs", 
                type="pdf", 
                accept_multiple_files=True,
                help="Puedes seleccionar varios archivos a la vez (Máx 50MB c/u)"
            )
            
            # El botón de envío está dentro del form
            submitted = st.form_submit_button("🚀 Procesar y Subir Archivos")
        
        # La lógica se ejecuta al enviar el formulario
        if submitted and uploaded_files:
            total_files = len(uploaded_files)
            main_progress = st.progress(0)
            status_text = st.empty()
            
            for i, pdf_file in enumerate(uploaded_files):
                status_text.text(f"Procesando archivo {i+1}/{total_files}: {pdf_file.name}...")
                
                try:
                    # A. Leer PDF
                    with st.spinner(f"Leyendo '{pdf_file.name}'..."):
                        bytes_data = pdf_file.getvalue()
                        reader = PdfReader(io.BytesIO(bytes_data))
                        text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                        
                        if not text:
                            st.warning(f"⚠️ El archivo {pdf_file.name} parece vacío o es una imagen.")
                            continue

                        docs, ids = process_text_to_docs(text, pdf_file.name)
                        total_vectors = len(docs)
                    
                    # B. Pinecone (CON BATCHING)
                    with st.spinner(f"Vectorizando {total_vectors} fragmentos..."):
                        try: vector_store.delete(filter={"source": pdf_file.name}) 
                        except: pass
                        
                        BATCH_SIZE = 100 
                        vec_progress = st.progress(0)
                        
                        for k in range(0, total_vectors, BATCH_SIZE):
                            batch_docs = docs[k : k + BATCH_SIZE]
                            batch_ids = ids[k : k + BATCH_SIZE]
                            vector_store.add_documents(batch_docs, ids=batch_ids)
                            vec_progress.progress(min((k + BATCH_SIZE) / total_vectors, 1.0))
                        
                        vec_progress.empty() 

                    # C. Storage
                    path = f"{username}/{sanitize_filename_to_ascii(pdf_file.name)}"
                    supabase_admin.storage.from_("manuales-pdf").upload(path, bytes_data, {"upsert": "true"})
                    
                    # D. SQL
                    supabase_admin.table('manuales').upsert({
                        'filename': pdf_file.name, 
                        'storage_path': path, 
                        'uploader_username': username, 
                        'vector_count': total_vectors
                    }, on_conflict='storage_path').execute()
                    
                    st.toast(f"✅ {pdf_file.name} subido correctamente.")

                except Exception as e:
                    st.error(f"❌ Error en '{pdf_file.name}': {e}")
                
                main_progress.progress((i + 1) / total_files)
            
            status_text.success(f"¡{total_files} manuales procesados!")
            st.cache_data.clear()
            
            # Mensaje final para el usuario
            st.info("La lista se ha limpiado. Puedes seleccionar nuevos archivos ahora.")

    # --- SECCIÓN 4: BIBLIOTECA DE MANUALES ---
    st.sidebar.divider()
    st.sidebar.subheader("📚 Biblioteca")
    try:
        res_manuales = supabase_admin.table('manuales').select('filename, storage_path').execute()
        mapa_manuales = {m['filename']: m['storage_path'] for m in res_manuales.data}
    except: mapa_manuales = {}

    if mapa_manuales:
        sel_manual = st.sidebar.selectbox("Descargar:", list(mapa_manuales.keys()), index=None, placeholder="Elige un manual...")
        if sel_manual:
            path_manual = mapa_manuales[sel_manual]
            if st.sidebar.button(f"📥 Obtener '{sel_manual}'"):
                try:
                    data_manual = supabase_admin.storage.from_("manuales-pdf").download(path_manual)
                    st.sidebar.download_button("💾 Guardar", data_manual, file_name=sel_manual, mime="application/pdf")
                except Exception as e: st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.caption("Biblioteca vacía.")

    # --- SECCIÓN 5: ADMIN PANEL ---
    if user_role == 'admin':
        render_admin_panel(username, credentials)

    # ==========================================
    # ÁREA PRINCIPAL DE CHAT
    # ==========================================
    st.title("🏦 Asistente Operacional")
    
    # Mensaje de bienvenida si no hay sesión activa
    if st.session_state.current_session_id is None:
        st.info("👋 ¡Hola! Escribe tu consulta abajo para iniciar un **Nuevo Chat**.")
    
    # Mostrar historial de mensajes de la sesión actual
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # --- INPUT DEL USUARIO ---
    if prompt := st.chat_input("Escribe tu consulta..."):
        
        # 1. CREAR SESIÓN SI NO EXISTE
        if st.session_state.current_session_id is None:
            # El primer mensaje se convierte en el título del chat
            new_id = create_new_session(username, prompt)
            if new_id:
                st.session_state.current_session_id = new_id
            else:
                st.error("Error al crear sesión de chat.")
                st.stop()

        # 2. GUARDAR Y MOSTRAR MENSAJE USUARIO
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        # Guardar en BD (usando la función de chat_service)
        save_message(username, st.session_state.current_session_id, "user", prompt)

        # 3. GENERAR RESPUESTA IA
        with st.chat_message("assistant"):
            with st.spinner("Analizando manuales..."):
                # Historial para LangChain (excluyendo el prompt actual para evitar duplicados en lógica interna)
                chat_history_obj = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" 
                    else AIMessage(content=m["content"]) 
                    for m in st.session_state.messages[:-1]
                ]

                # Invocar RAG
                response = rag_chain.invoke({
                    "input": prompt, 
                    "chat_history": chat_history_obj
                })
                
                answer = response["answer"]
                context_docs = response.get("context", [])

                # Mostrar respuesta
                st.markdown(answer)
                
                # Mostrar Contexto (Fuentes)
                with st.expander("🔍 Ver contexto recuperado (Fuentes)", expanded=False):
                    if context_docs:
                        for i, doc in enumerate(context_docs):
                            st.markdown(f"**Fragmento {i+1}** | *Fuente: {doc.metadata.get('source', 'Desconocido')}*")
                            st.caption(doc.page_content)
                            st.divider()
                    else:
                        st.warning("No se encontró contexto en los manuales. Respuesta generada con conocimiento general.")

                # 4. GUARDAR MENSAJE IA
                ai_msg = {"role": "assistant", "content": answer}
                st.session_state.messages.append(ai_msg)
                save_message(username, st.session_state.current_session_id, "assistant", answer)