# modules/utils.py
import re
import hashlib
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def sanitize_filename_to_ascii(filename):
    """Limpia nombres de archivo para evitar errores en IDs."""
    replacements = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', ' ': '_'}
    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    return filename

def process_text_to_docs(text, source_name):
    """
    Convierte texto crudo en documentos listos para vectorizar.
    CAMBIO IMPORTANTE: Aumentamos chunk_size para capturar procedimientos completos.
    """
    new_doc = Document(page_content=text, metadata={"source": source_name})
    
    # ANTES: chunk_size=512, chunk_overlap=20
    # AHORA: chunk_size=1500, chunk_overlap=300
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""] # Intenta cortar primero por párrafos
    )
    
    docs = text_splitter.split_documents([new_doc])
    
    ids = []
    filename_ascii = sanitize_filename_to_ascii(source_name)
    for i, doc in enumerate(docs):
        # Hash del contenido para asegurar unicidad
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        chunk_id = f"{filename_ascii}_{content_hash}_{i}"
        ids.append(chunk_id)
        
    return docs, ids