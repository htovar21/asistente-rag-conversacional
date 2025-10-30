import os
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
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Requerido por el cliente base
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag")
DOCUMENT_DIR = "manuales"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 # Dimensión para all-MiniLM-L6-v2

# --- 3. Inicialización y Verificación de Pinecone ---
try:
    print("Iniciando conexión con Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = pc.list_indexes()
    index_names = [index_info['name'] for index_info in existing_indexes]
    
    if PINECONE_INDEX_NAME not in index_names: 
        print(f"🚨 ERROR: El índice '{PINECONE_INDEX_NAME}' NO EXISTE en Pinecone.")
        print(f"Por favor, créalo manualmente con {EMBEDDING_DIMENSION} dimensiones.")
        exit(1)
    else:
        print(f"✅ Conexión establecida con el índice '{PINECONE_INDEX_NAME}'.")
        # Verificación de dimensión
        index_stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
        if index_stats.dimension != EMBEDDING_DIMENSION:
            print(f"🚨 ERROR: La dimensión del índice ({index_stats.dimension}) no coincide con la requerida ({EMBEDDING_DIMENSION}).")
            print("Por favor, bórrelo y créelo con 384 dimensiones.")
            exit(1)
        print(f"✅ Dimensión del índice ({EMBEDDING_DIMENSION}) confirmada.")

except Exception as e:
    print(f"🚨 ERROR FATAL: No se pudo conectar a Pinecone.")
    print(f"Revisa tu PINECONE_API_KEY. Detalle: {e}")
    exit(1)

# --- 4. Implementación del Pipeline de Ingesta (LangChain) ---
try:
    # Carga los documentos de la carpeta 'manuales/' (solo PDFs)
    print(f"\nCargando documentos PDF desde la carpeta '{DOCUMENT_DIR}'...")
    loader = DirectoryLoader(
        DOCUMENT_DIR,
        glob="**/*.pdf", # Carga solo archivos .pdf
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    
    if not documents:
        print(f"No se encontraron documentos PDF en '{DOCUMENT_DIR}'. Saliendo.")
        exit(0)
    print(f"Se encontraron {len(documents)} documentos PDF.")

    # Configura el segmentador de texto (chunker)
    # Usamos RecursiveCharacterTextSplitter, que es robusto en LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, # Mantenemos el tamaño del script LlamaIndex
        chunk_overlap=20  # Mantenemos el solapamiento
    )
    docs = text_splitter.split_documents(documents)
    print(f"Documentos segmentados en {len(docs)} fragmentos (chunks).")

    # Inicializa el modelo de embeddings
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Forzar CPU
    )
    print("Modelo de embeddings cargado.")

    # Sube los documentos a Pinecone
    print("Iniciando carga de vectores a Pinecone (esto puede tardar)...")
    # PineconeVectorStore.from_documents se encarga de conectar y subir
    PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=PINECONE_INDEX_NAME
        # El cliente de LangChain leerá la API Key desde las variables de entorno
    )

    print("\n========================================================")
    print("🎉 Ingesta de Documentos Completada con Éxito (usando LangChain).")
    print(f"Se indexaron {len(docs)} fragmentos en Pinecone.")
    print("========================================================")

except Exception as e:
    print(f"🚨 ERROR en la fase de Ingesta de Datos (LangChain).")
    print(f"Detalle del error: {e}")