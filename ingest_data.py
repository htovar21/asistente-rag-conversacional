import os
import hashlib
import re
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 1. Carga de Variables de Entorno ---
load_dotenv()

# --- 2. Variables del Proyecto ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag")
DOCUMENT_DIR = "manuales"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 100

# --- Función para limpiar IDs (ASCII) ---
def sanitize_filename_to_ascii(filename):
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', ' ': '_'
    }
    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    return filename

# --- INICIO DEL SCRIPT ---
print("========================================================")
print("INICIANDO SCRIPT DE INGESTA (LangChain + Pinecone)")
print("========================================================")

try:
    # --- PASO 1/6: Conectar y Verificar Pinecone ---
    print("\n--- PASO 1/6: Conectando y verificando Pinecone ---")
    start_time = time.time()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    index_names = [index_info['name'] for index_info in existing_indexes]
    
    if PINECONE_INDEX_NAME not in index_names: 
        print(f"🚨 ERROR: El índice '{PINECONE_INDEX_NAME}' NO EXISTE en Pinecone.")
        print(f"Por favor, créalo manualmente con {EMBEDDING_DIMENSION} dimensiones.")
        exit(1)
    
    print(f"✅ Conexión establecida con el índice '{PINECONE_INDEX_NAME}'.")
    index_stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
    
    if index_stats.dimension != EMBEDDING_DIMENSION:
        print(f"🚨 ERROR: Dimensión del índice ({index_stats.dimension}) no coincide.")
        exit(1)
        
    print(f"✅ Dimensión del índice ({EMBEDDING_DIMENSION}) confirmada.")
    print(f"  (Tiempo: {time.time() - start_time:.2f}s)")

    # --- PASO 2/6: Extracción de Documentos ---
    print("\n--- PASO 2/6: Extrayendo Documentos (Lectura) ---")
    start_time = time.time()
    
    loader = DirectoryLoader(
        DOCUMENT_DIR,
        glob="**/*.pdf", # Carga solo archivos .pdf
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True # Ignora PDFs corruptos
    )
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    
    if not documents:
        print(f"🚨 ERROR: No se encontraron documentos PDF legibles en '{DOCUMENT_DIR}'. Saliendo.")
        exit(0)
        
    print(f"✅ Se encontraron y cargaron {len(documents)} páginas de {DOCUMENT_DIR}.")
    print(f"  (Tiempo: {time.time() - start_time:.2f}s)")

    # --- PASO 3/6: Transformación (Segmentación) ---
    print("\n--- PASO 3/6: Transformando Documentos (Segmentación) ---")
    start_time = time.time()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20
    )
    docs = text_splitter.split_documents(documents)
    
    print(f"✅ Documentos segmentados en {len(docs)} fragmentos (chunks).")
    print(f"  (Tiempo: {time.time() - start_time:.2f}s)")

    # --- PASO 4/6: Cargar Modelo de Embeddings ---
    print("\n--- PASO 4/6: Cargando Modelo de Embeddings ---")
    start_time = time.time()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"✅ Modelo '{EMBEDDING_MODEL_NAME}' cargado en memoria.")
    print(f"  (Tiempo: {time.time() - start_time:.2f}s)")

    # --- PASO 5/6: Generación de IDs Únicos (Basado en Fuente + Índice) ---
    print("\n--- PASO 5/6: Generando IDs únicos (para evitar duplicados) ---")
    start_time = time.time()
    ids = []
    
    # --- CAMBIO DE LÓGICA ---
    # Usamos enumerate() para obtener un índice (i) para cada fragmento
    for i, doc in enumerate(docs): 
        source = doc.metadata.get("source", "unknown")
        
        # Limpiamos el nombre del archivo fuente
        filename = os.path.basename(source)
        filename_ascii = sanitize_filename_to_ascii(filename)
        
        # El ID final es una combinación de la fuente y el NÚMERO del fragmento
        chunk_id = f"{filename_ascii}_chunk_{i}"
        
        ids.append(chunk_id)

    if len(ids) != len(docs):
        print("🚨 ERROR: El conteo de IDs no coincide con el conteo de fragmentos.")
        exit(1)
    
    print(f"✅ Se generaron {len(ids)} IDs únicos.")
    print(f"  (Tiempo: {time.time() - start_time:.2f}s)")
    
    # --- PASO 6/6: Carga (Upsert) a Pinecone en Lotes ---
    print(f"\n--- PASO 6/6: Cargando Vectores a Pinecone (en lotes de {BATCH_SIZE}) ---")
    start_time = time.time()
    
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    total_lotes = (len(docs) // BATCH_SIZE) + (1 if len(docs) % BATCH_SIZE > 0 else 0)
    
    for i in range(0, len(docs), BATCH_SIZE):
        lote_actual = i // BATCH_SIZE + 1
        print(f"  Subiendo Lote {lote_actual}/{total_lotes} ({len(docs[i:i+BATCH_SIZE])} fragmentos)...")
        
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        
        vector_store.add_documents(batch_docs, ids=batch_ids)
        
    print(f"✅ Carga (Upsert) a Pinecone completada.")
    print(f"  (Tiempo: {time.time() - start_time:.2f}s)")

    print("\n========================================================")
    print("🎉 INGESTA (UPSERT) COMPLETADA CON ÉXITO.")
    print(f"Se verificaron/actualizaron {len(docs)} fragmentos en Pinecone.")
    print("========================================================")

except Exception as e:
    print(f"🚨 ERROR en la fase de Ingesta de Datos (LangChain).")
    print(f"Detalle del error: {e}")