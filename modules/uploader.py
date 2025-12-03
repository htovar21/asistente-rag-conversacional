import streamlit as st
import io
from pypdf import PdfReader
from modules.database import supabase_admin
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii

def render_upload_section(username, vector_store, key_suffix=""):
    """
    Renderiza el botón y la lógica de subida de archivos.
    key_suffix: Sirve para que Streamlit no confunda el uploader del sidebar con el del modal.
    """
    # Usamos un formulario
    with st.form(f"upload_form_{key_suffix}", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Seleccionar PDFs", 
            type="pdf", 
            accept_multiple_files=True,
            key=f"file_uploader_{key_suffix}"
        )
        submitted = st.form_submit_button("🚀 Procesar y Subir", type="primary")
    
    if submitted and uploaded_files:
        total_files = len(uploaded_files)
        main_progress = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(uploaded_files):
            status_text.text(f"Procesando {i+1}/{total_files}: {pdf_file.name}...")
            
            try:
                # 1. Calcular Peso
                bytes_data = pdf_file.getvalue()
                file_size_bytes = len(bytes_data)

                # 2. Leer Texto
                reader = PdfReader(io.BytesIO(bytes_data))
                text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                
                if not text:
                    st.warning(f"⚠️ '{pdf_file.name}' parece vacío.")
                    continue

                docs, ids = process_text_to_docs(text, pdf_file.name)
                total_vectors = len(docs)
                
                # 3. Batching a Pinecone
                with st.spinner(f"Indexando '{pdf_file.name}'..."):
                    try: vector_store.delete(filter={"source": pdf_file.name}) 
                    except: pass
                    
                    BATCH_SIZE = 100 
                    for k in range(0, total_vectors, BATCH_SIZE):
                        batch_docs = docs[k : k + BATCH_SIZE]
                        batch_ids = ids[k : k + BATCH_SIZE]
                        vector_store.add_documents(batch_docs, ids=batch_ids)

                # 4. Storage & SQL
                path = f"{username}/{sanitize_filename_to_ascii(pdf_file.name)}"
                supabase_admin.storage.from_("manuales-pdf").upload(path, bytes_data, {"upsert": "true"})
                
                supabase_admin.table('manuales').upsert({
                    'filename': pdf_file.name, 
                    'storage_path': path, 
                    'uploader_username': username, 
                    'vector_count': total_vectors,
                    'file_size': file_size_bytes # Guardamos el peso
                }, on_conflict='storage_path').execute()
                
                st.toast(f"✅ {pdf_file.name} subido.")

            except Exception as e:
                st.error(f"Error en '{pdf_file.name}': {e}")
            
            main_progress.progress((i + 1) / total_files)
        
        status_text.success("¡Carga completa!")
        st.cache_data.clear()