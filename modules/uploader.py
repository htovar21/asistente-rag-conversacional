import streamlit as st
import io
import time
from pypdf import PdfReader
from modules.database import supabase_admin
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii

# --- FUNCIÓN LÓGICA DE PROCESAMIENTO ---
def execute_upload_process(files, username, vector_store):
    """Ejecuta la lectura, vectorización y subida con feedback visual."""
    
    # Validamos sesión para obtener el ID numérico
    user_id = st.session_state.get('user_id')
    
    if not user_id:
        st.error("Error de sesión: No se encontró el ID del usuario. Por favor, relogueate.")
        return

    total_files = len(files)
    
    status_container = st.empty()
    main_progress_bar = st.progress(0)
    
    for i, pdf_file in enumerate(files):
        # Limpieza visual y reset del puntero del archivo
        pdf_file.seek(0)
        status_container.empty()
        time.sleep(0.1)
        
        main_progress_bar.progress(i / total_files, text=f"Progreso Total: {i} de {total_files} completados")
        
        with status_container.container():
            with st.container(border=True):
                st.info(f"### Procesando archivo {i+1} de {total_files}\n**Documento:** `{pdf_file.name}`", icon="📂")
                file_progress_bar = st.progress(0, text="Iniciando lectura...")
                
                try:
                    # 1. Lectura
                    start_time = time.time()
                    file_progress_bar.progress(10, text="Extrayendo texto...")
                    bytes_data = pdf_file.getvalue()
                    file_size_bytes = len(bytes_data)
                    reader = PdfReader(io.BytesIO(bytes_data))
                    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    
                    if not text:
                        st.warning(f"El archivo **{pdf_file.name}** parece vacío o ilegible.")
                        time.sleep(2)
                        continue

                    # 2. Preparación
                    file_progress_bar.progress(25, text="Fragmentando contenido...")
                    docs, ids = process_text_to_docs(text, pdf_file.name)
                    total_vectors = len(docs)

                    # 3. Limpieza previa (Borrar vectores viejos si existen)
                    # Esto asegura que la IA no mezcle la versión vieja con la nueva
                    try: vector_store.delete(filter={"source": pdf_file.name}) 
                    except: pass

                    # 4. Indexación
                    BATCH_SIZE = 100 
                    total_batches = (total_vectors // BATCH_SIZE) + 1
                    
                    if total_batches == 1:
                        file_progress_bar.progress(40, text="Memorizando...")
                        time.sleep(0.2) 

                    for idx, k in enumerate(range(0, total_vectors, BATCH_SIZE)):
                        batch_docs = docs[k : k + BATCH_SIZE]
                        batch_ids = ids[k : k + BATCH_SIZE]
                        vector_store.add_documents(batch_docs, ids=batch_ids)
                        
                        batch_progress = (idx + 1) / total_batches
                        visual_progress = 30 + int(batch_progress * 50)
                        file_progress_bar.progress(visual_progress, text=f"Memorizando... ({min(k + BATCH_SIZE, total_vectors)}/{total_vectors} vecs)")

                    # 5. Storage (Supabase)
                    file_progress_bar.progress(85, text="Subiendo archivo...")
                    
                    # --- CORRECCIÓN APLICADA AQUÍ ---
                    # Usamos una carpeta común 'biblioteca/' para forzar la misma ruta
                    # sin importar qué usuario suba el archivo.
                    path = f"biblioteca/{sanitize_filename_to_ascii(pdf_file.name)}"
                    
                    supabase_admin.storage.from_("manuales-pdf").upload(
                        path=path, 
                        file=bytes_data, 
                        file_options={"upsert": "true", "content-type": "application/pdf"}
                    )
                    
                    # 6. SQL Metadata
                    file_progress_bar.progress(95, text=" Registrando...")
                    
                    # Al coincidir el 'storage_path', se actualiza el registro existente
                    # cambiando el 'uploader_id' al usuario actual.
                    supabase_admin.table('manuales').upsert({
                        'filename': pdf_file.name, 
                        'storage_path': path, 
                        'uploader_id': user_id,
                        'vector_count': total_vectors,
                        'file_size': file_size_bytes
                    }, on_conflict='storage_path').execute()
                    
                    elapsed = round(time.time() - start_time, 1)
                    file_progress_bar.progress(100, text="✨ ¡Completado!")
                    st.success(f" **{pdf_file.name}** procesado en {elapsed}s.")
                    time.sleep(1) 

                except Exception as e:
                    st.error(f"Error crítico en **{pdf_file.name}**: {e}")
                    time.sleep(4)
    
    main_progress_bar.progress(100, text=f"¡Operación Finalizada!")
    status_container.success("**Todos los documentos han sido procesados correctamente.**")
    time.sleep(1.5)
    status_container.empty()
    main_progress_bar.empty()


# --- DIALOG PARA MODO SIDEBAR ---
@st.dialog("Archivos Duplicados Detectados")
def confirm_dialog_modal(existing_files, all_files, username, vector_store, state_key_name):
    st.warning("Los siguientes archivos ya existen y serán reemplazados:")
    for f in existing_files:
        st.markdown(f"- 📄 `{f}`")
    
    st.write("---")
    c1, c2 = st.columns(2)
    if c1.button("Cancelar", key="dlg_cancel"): st.rerun()
    
    if c2.button("Sobreescribir", key="dlg_confirm", type="primary"):
        execute_upload_process(all_files, username, vector_store)
        # Limpiamos el uploader tras confirmar
        if state_key_name in st.session_state:
            st.session_state[state_key_name] += 1
        st.rerun()


# --- COMPONENTE PRINCIPAL ---
def render_upload_section(username, vector_store, key_suffix="", use_modal=True):
    
    # Estados
    state_pending = f"pending_{key_suffix}"
    state_key = f"uploader_key_{key_suffix}"

    if state_key not in st.session_state: st.session_state[state_key] = 0

    # 1. MOSTRAR CONFIRMACIÓN INLINE (Si hay pendientes y no es modal)
    if not use_modal and state_pending in st.session_state:
        package = st.session_state[state_pending] # (files, existing_names)
        files = package['files']
        existing = package['existing']
        
        with st.container(border=True):
            st.warning("**Atención: Archivos Duplicados**")
            st.caption("Los siguientes archivos ya existen en la biblioteca:")
            for f in existing: st.markdown(f"- `{f}`")
            
            st.write("**¿Deseas sobreescribirlos?**")
            c1, c2 = st.columns(2)
            
            if c1.button("Cancelar Operación", key=f"inl_cancel_{key_suffix}", use_container_width=True):
                del st.session_state[state_pending]
                st.rerun()
            
            if c2.button("Sí, Sobreescribir", key=f"inl_confirm_{key_suffix}", type="primary", use_container_width=True):
                execute_upload_process(files, username, vector_store)
                del st.session_state[state_pending]
                st.session_state[state_key] += 1 # Limpia el uploader
                st.rerun()
        return 

    # 2. FORMULARIO DE SUBIDA
    current_key_id = st.session_state[state_key]
    
    with st.form(f"form_{key_suffix}_{current_key_id}"):
        files = st.file_uploader(
            "Seleccionar PDFs", type="pdf", accept_multiple_files=True,
            key=f"file_{key_suffix}_{current_key_id}",
            help="Máximo 50MB."
        )
        submit = st.form_submit_button("Iniciar Carga", type="primary")
    
    # LÓGICA
    if submit and files:
        filenames = [f.name for f in files]
        try:
            res = supabase_admin.table('manuales').select('filename').in_('filename', filenames).execute()
            existing = [r['filename'] for r in res.data]
            
            if existing:
                if use_modal:
                    confirm_dialog_modal(existing, files, username, vector_store, state_key)
                else:
                    # Modo Inline
                    st.session_state[state_pending] = {'files': files, 'existing': existing}
                    st.rerun()
            else:
                # Flujo normal sin duplicados
                execute_upload_process(files, username, vector_store)
                st.session_state[state_key] += 1
                st.rerun()
                
        except Exception as e: st.error(f"Error: {e}")