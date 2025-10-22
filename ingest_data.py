import os
from dotenv import load_dotenv
# Se remueve ServerlessSpec porque no la necesitamos para la conexión ni la creación
from pinecone import Pinecone 
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext

# --- 1. Carga de Variables de Entorno ---
load_dotenv()

# --- 2. Variables del Proyecto ---
# Pinecone: Usamos las claves de .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Debe ser 'us-east-1'
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag") 
DOCUMENT_DIR = "manuales"

# Hugging Face: Modelo de embeddings gratuito (384 DIMENSIONES)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 

# --- 3. Inicialización y Conexión (Corregido) ---
try:
    print("Iniciando conexión con Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    # --- CORRECCIÓN: Obtener la lista de nombres de índices ---
    # La versión moderna de Pinecone devuelve una lista de objetos IndexModel,
    # por lo que extraemos los nombres.
    
    # Obtener todos los índices existentes
    existing_indexes = pc.list_indexes()
    
    # Extraer los nombres de los índices existentes
    index_names = [index_info['name'] for index_info in existing_indexes]
    
    # Verificar si el índice existe en la lista de nombres
    if PINECONE_INDEX_NAME not in index_names: 
        print(f"🚨 ERROR: El índice '{PINECONE_INDEX_NAME}' NO EXISTE en Pinecone.")
        print("Por favor, créalo manualmente con 384 dimensiones.")
        exit(1) # Detiene el script si el índice no existe
    else:
        print(f"✅ Conexión establecida con el índice '{PINECONE_INDEX_NAME}'.")

except Exception as e:
    print(f"🚨 ERROR FATAL: No se pudo conectar a Pinecone.")
    print(f"Revisa tu PINECONE_API_KEY y la conexión de red. Detalle: {e}")
    exit(1) # Detiene el script si la conexión falla

# --- 4. Configuración del Embedder y Vector Store ---

# Conecta LlamaIndex al índice Pinecone existente
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Inicializa el modelo de embeddings
embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    device="cpu" # Lo ejecutamos en la CPU de tu PC
)
print(f"Modelo de embeddings '{EMBEDDING_MODEL_NAME}' cargado localmente ({EMBEDDING_DIMENSION} dim).")

# Prepara el Storage Context para que LlamaIndex use Pinecone
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# --- 5. Implementación del Pipeline de Ingesta ---

try:
    # Carga los documentos de la carpeta 'manuales/'
    print(f"\nCargando documentos desde la carpeta '{DOCUMENT_DIR}/'...")
    documents = SimpleDirectoryReader(DOCUMENT_DIR).load_data()
    print(f"Se encontraron {len(documents)} documentos para procesar.")

    # Configura el segmentador de texto (chunker)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20) 

    # Define el Pipeline: Lectura -> Split -> Embed -> Store
    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            embed_model, # Usa el embedder de Hugging Face
        ],
        vector_store=vector_store,
    )

    print("Iniciando ingesta: segmentando, creando vectores y subiendo a Pinecone...")
    
    # Ejecuta el pipeline
    nodes = pipeline.run(documents=documents)

    print("\n========================================================")
    print("🎉 Ingesta de Documentos Completada con Éxito.")
    print(f"Documentos indexados en Pinecone: {len(nodes)} fragmentos (chunks).")
    print("¡Ya puedes desarrollar tu aplicación web!")
    print("========================================================")

except Exception as e:
    print(f"🚨 ERROR en la fase de Ingesta de Datos.")
    print("Verifica si tus archivos en 'manuales/' son legibles (ej: PDFs válidos).")
    print(f"Detalle del error: {e}")