import streamlit as st
import io
import time
from pypdf import PdfReader
from modules.database import supabase_admin
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii

def render_upload_section(username, vector_store, key_suffix=""):
    """
    Módulo de carga simplificado (Compatible con Sidebar):
    Usa barras de progreso estándar en lugar de st.status para evitar errores de anidación.
    """
    with st.form(f"upload_form_{key_suffix}", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Seleccionar PDFs", 
            type="pdf", 
            accept_multiple_files=True,
            key=f"file_uploader_{key_suffix}",
            help="Soporta carga por lotes. Máximo 50MB por archivo."
        )
        submitted = st.form_submit_button("🚀 Iniciar Carga", type="primary")
    
    if submitted and uploaded_files:
        total_files = len(uploaded_files)
        
        # Elementos de UI para feedback (Placeholder)
        main_bar = st.progress(0, text="Iniciando operación...")
        status_text = st.empty() # Contenedor dinámico para mensajes
        
        for i, pdf_file in enumerate(uploaded_files):
            # Notificar inicio del archivo actual
            status_text.info(f"🔄 Procesando **{pdf_file.name}** ({i+1}/{total_files})...")
            
            try:
                # --- PASO 1: LECTURA ---
                start_time = time.time()
                bytes_data = pdf_file.getvalue()
                file_size_bytes = len(bytes_data)
                
                reader = PdfReader(io.BytesIO(bytes_data))
                text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                
                if not text:
                    st.toast(f"⚠️ Archivo vacío o imagen: {pdf_file.name}", icon="⚠️")
                    continue

                # --- PASO 2: PREPARACIÓN ---
                docs, ids = process_text_to_docs(text, pdf_file.name)
                total_vectors = len(docs)

                # --- PASO 3: INDEXACIÓN (BATCHING) ---
                # Limpieza previa
                try: vector_store.delete(filter={"source": pdf_file.name}) 
                except: pass
                
                BATCH_SIZE = 100 
                
                for k in range(0, total_vectors, BATCH_SIZE):
                    batch_docs = docs[k : k + BATCH_SIZE]
                    batch_ids = ids[k : k + BATCH_SIZE]
                    vector_store.add_documents(batch_docs, ids=batch_ids)
                    
                    # Feedback detallado del progreso de vectorización
                    porcentaje = int(min((k + BATCH_SIZE) / total_vectors, 1.0) * 100)
                    status_text.caption(f"🧠 Vectorizando **{pdf_file.name}**: {porcentaje}% completado")

                # --- PASO 4: ALMACENAMIENTO ---
                status_text.caption(f"☁️ Subiendo **{pdf_file.name}** a la nube...")
                
                path = f"{username}/{sanitize_filename_to_ascii(pdf_file.name)}"
                
                supabase_admin.storage.from_("manuales-pdf").upload(
                    path=path, 
                    file=bytes_data, 
                    file_options={
                        "upsert": "true", 
                        "content-type": "application/pdf" # Mantenemos el fix del JSON
                    }
                )
                
                # --- PASO 5: REGISTRO SQL ---
                supabase_admin.table('manuales').upsert({
                    'filename': pdf_file.name, 
                    'storage_path': path, 
                    'uploader_username': username, 
                    'vector_count': total_vectors,
                    'file_size': file_size_bytes
                }, on_conflict='storage_path').execute()
                
                elapsed = round(time.time() - start_time, 1)
                st.toast(f"✅ {pdf_file.name} listo ({elapsed}s).", icon="✅")

            except Exception as e:
                st.error(f"❌ Error en '{pdf_file.name}': {e}")
                print(f"Error detalle: {e}")
        
            # Actualizar barra GENERAL
            main_bar.progress((i + 1) / total_files, text=f"Progreso Total: {i+1} de {total_files}")

        # Finalización
        status_text.success("✅ ¡Carga Completada!")
        time.sleep(2)
        main_bar.empty()
        status_text.empty()
        st.cache_data.clear()