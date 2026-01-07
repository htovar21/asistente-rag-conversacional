import os
import io
from pypdf import PdfReader
from modules.database import supabase_admin, pinecone_index
from modules.utils import process_text_to_docs, sanitize_filename_to_ascii
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from modules.config import CONFIG

# Configuración del Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = PineconeVectorStore(index_name=CONFIG["PINECONE_INDEX_NAME"], embedding=embeddings)

# ID del administrador en tu tabla 'usuarios' (ajusta si no es el 5)
UPLOADER_ID_DEFAULT = 5 

def bulk_upload():
    folder_path = "manuales" 
    
    if not os.path.exists(folder_path):
        print(f" Crea la carpeta '{folder_path}' y pon los PDFs ahí.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f" Encontrados {len(files)} manuales. Iniciando carga masiva...")

    for i, filename in enumerate(files):
        print(f"[{i+1}/{len(files)}] Procesando: {filename}...")
        
        try:
            file_path = os.path.join(folder_path, filename)
            
            # 1. Leer PDF Local
            with open(file_path, "rb") as f:
                bytes_data = f.read()
                
            reader = PdfReader(io.BytesIO(bytes_data))
            text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
            
            if not text:
                print(f"    PDF Vacío o imagen: {filename}")
                continue

            # 2. Vectorización
            docs, ids = process_text_to_docs(text, filename)
            
            # Limpieza de vectores previos para evitar duplicados
            try: 
                vector_store.delete(filter={"source": filename})
            except: 
                pass
            
            vector_store.add_documents(docs, ids=ids)
            
            # 3. Subir a Supabase Storage (CAMBIO AQUÍ: Carpeta biblioteca/)
            # Se usa 'biblioteca/' para organizar los archivos físicamente
            storage_path = f"biblioteca/{sanitize_filename_to_ascii(filename)}"
            
            supabase_admin.storage.from_("manuales-pdf").upload(
                path=storage_path, 
                file=bytes_data, 
                file_options={"upsert": "true"}
            )
            
            # 4. Registrar metadatos en SQL
            # 'storage_path' es la clave única para evitar registros duplicados
            supabase_admin.table('manuales').upsert({
                'filename': filename, 
                'storage_path': storage_path, 
                'uploader_id': UPLOADER_ID_DEFAULT, 
                'vector_count': len(docs),
                'file_size': len(bytes_data)
            }, on_conflict='storage_path').execute()
            
            print(f"    Listo ({len(docs)} vectores).")

        except Exception as e:
            print(f"    Error: {e}")

    print("\n Carga masiva completada.")

if __name__ == "__main__":
    bulk_upload()