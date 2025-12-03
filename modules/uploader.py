import streamlit as st
import io
import time
from pypdf import PdfReader
from modules.database import supabase_admin
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii

# --- FUNCIÓN LÓGICA DE PROCESAMIENTO (VISUALMENTE MEJORADA) ---
def execute_upload_process(files, username, vector_store):
    """Ejecuta la lectura, vectorización y subida con feedback visual detallado."""
    
    total_files = len(files)
    
    # Contenedores fijos para que la interfaz no salte
    status_container = st.empty()
    main_progress_bar = st.progress(0)
    
    for i, pdf_file in enumerate(files):
        # Actualizar barra general
        general_progress = i / total_files
        main_progress_bar.progress(general_progress, text=f"Progreso Total: {i}/{total_files} archivos completados")
        
        # Tarjeta de Estado Grande y Clara
        with status_container.container():
            st.info(f"### 🔄 Procesando archivo {i+1} de {total_files}\n**Documento:** `{pdf_file.name}`", icon="📂")
            
            # Barra de progreso específica para EL ARCHIVO ACTUAL
            file_progress_bar = st.progress(0, text="Iniciando lectura...")
            
            try:
                # 1. Lectura
                start_time = time.time()
                file_progress_bar.progress(10, text="📖 Extrayendo texto del PDF...")
                
                bytes_data = pdf_file.getvalue()
                file_size_bytes = len(bytes_data)
                
                reader = PdfReader(io.BytesIO(bytes_data))
                text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                
                if not text:
                    st.warning(f"⚠️ El archivo **{pdf_file.name}** parece estar vacío o es una imagen.")
                    time.sleep(2)
                    continue

                # 2. Preparación
                file_progress_bar.progress(25, text="🧩 Fragmentando contenido en vectores...")
                docs, ids = process_text_to_docs(text, pdf_file.name)
                total_vectors = len(docs)

                # 3. Limpieza previa
                try: vector_store.delete(filter={"source": pdf_file.name}) 
                except: pass

                # 4. Indexación (Batching) - AQUÍ ES DONDE ANTES SE CONGELABA
                BATCH_SIZE = 100 
                total_batches = (total_vectors // BATCH_SIZE) + 1
                
                for idx, k in enumerate(range(0, total_vectors, BATCH_SIZE)):
                    batch_docs = docs[k : k + BATCH_SIZE]
                    batch_ids = ids[k : k + BATCH_SIZE]
                    vector_store.add_documents(batch_docs, ids=batch_ids)
                    
                    # Actualización en TIEMPO REAL de la barra
                    # Mapeamos el progreso del lote (0-100%) al rango visual de la barra (30% - 80%)
                    batch_progress = (idx + 1) / total_batches
                    visual_progress = 30 + int(batch_progress * 50)
                    
                    file_progress_bar.progress(
                        visual_progress, 
                        text=f"🧠 Memorizando... ({min(k + BATCH_SIZE, total_vectors)} / {total_vectors} vectores)"
                    )

                # 5. Storage
                file_progress_bar.progress(85, text="☁️ Subiendo archivo a la nube segura...")
                path = f"{username}/{sanitize_filename_to_ascii(pdf_file.name)}"
                
                supabase_admin.storage.from_("manuales-pdf").upload(
                    path=path, 
                    file=bytes_data, 
                    file_options={"upsert": "true", "content-type": "application/pdf"}
                )
                
                # 6. SQL
                file_progress_bar.progress(95, text="💾 Guardando registro en base de datos...")
                supabase_admin.table('manuales').upsert({
                    'filename': pdf_file.name, 
                    'storage_path': path, 
                    'uploader_username': username, 
                    'vector_count': total_vectors,
                    'file_size': file_size_bytes
                }, on_conflict='storage_path').execute()
                
                elapsed = round(time.time() - start_time, 1)
                
                # Finalización visual del archivo
                file_progress_bar.progress(100, text="✨ ¡Completado!")
                st.toast(f"✅ {pdf_file.name} listo ({elapsed}s).", icon="✅")
                time.sleep(0.5) 

            except Exception as e:
                st.error(f"❌ Error crítico en **{pdf_file.name}**: {e}")
                time.sleep(3) # Pausa para leer el error
    
    # Finalización Total
    main_progress_bar.progress(100, text="¡Operación Finalizada!")
    status_container.success("✅ **Todos los documentos han sido procesados correctamente.**")
    time.sleep(2)
    
    # Limpieza
    status_container.empty()
    main_progress_bar.empty()
    st.cache_data.clear()


# --- MODAL DE CONFIRMACIÓN ---
@st.dialog("⚠️ Archivos Duplicados Detectados")
def confirm_overwrite_modal(existing_files, all_files, username, vector_store):
    st.warning("Los siguientes archivos ya existen en la base de conocimiento:")
    
    for f in existing_files:
        st.markdown(f"- 📄 **{f}**")
        
    st.write("---")
    st.markdown("### ¿Deseas sobreescribirlos?")
    st.caption("Esta acción actualizará el contenido en la memoria de la IA.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Cancelar", use_container_width=True):
            st.rerun() 
            
    with col2:
        if st.button("✅ Sí, Sobreescribir", type="primary", use_container_width=True):
            # Usamos la misma función visual mejorada
            execute_upload_process(all_files, username, vector_store)
            st.rerun()


# --- COMPONENTE PRINCIPAL ---
def render_upload_section(username, vector_store, key_suffix=""):
    
    if f"uploader_key_{key_suffix}" not in st.session_state:
        st.session_state[f"uploader_key_{key_suffix}"] = 0

    dynamic_key = f"file_uploader_{key_suffix}_{st.session_state[f'uploader_key_{key_suffix}']}"

    with st.form(f"upload_form_{key_suffix}"):
        uploaded_files = st.file_uploader(
            "Seleccionar PDFs", 
            type="pdf", 
            accept_multiple_files=True,
            key=dynamic_key,
            help="Máximo 50MB. Si el archivo existe, te pedirá confirmación."
        )
        submitted = st.form_submit_button("🚀 Iniciar Carga", type="primary")
    
    if submitted and uploaded_files:
        filenames = [f.name for f in uploaded_files]
        
        try:
            res = supabase_admin.table('manuales').select('filename').in_('filename', filenames).execute()
            existing_db_files = [row['filename'] for row in res.data]
            
            if existing_db_files:
                confirm_overwrite_modal(existing_db_files, uploaded_files, username, vector_store)
                st.session_state[f"uploader_key_{key_suffix}"] += 1
            else:
                # Usamos la función visual mejorada directamente
                execute_upload_process(uploaded_files, username, vector_store)
                
                st.session_state[f"uploader_key_{key_suffix}"] += 1
                st.rerun()
                
        except Exception as e:
            st.error(f"Error de verificación: {e}")