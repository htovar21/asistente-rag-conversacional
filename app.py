import streamlit as st
import os
import pinecone # Importar pinecone base

# --- Importaciones de LangChain ---
# CAMBIO: Usar cargador de directorio en lugar de archivo único si fuera necesario indexar de nuevo
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter # Mantenemos el splitter si se reindexara
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore # <-- CONECTOR PINECONE PARA LANGCHAIN
# from langchain_community.vectorstores import FAISS # Ya no usamos FAISS
# from langchain_core.vectorstores import VectorStoreRetriever # PineconeVectorStore actúa como retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage # Para el historial de chat

# --- Configuración Inicial ---
st.set_page_config(
    page_title="Agente RAG Bancario (LangChain)",
    page_icon="🏦",
    layout="centered" # Cambiado a centered para un chat más típico
)
st.title("🏦 Agente RAG Bancario (con LangChain)")
st.caption("Asistente impulsado por la documentación interna del Banco Caroní.")

# --- Carga de Secretos y Configuración ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # LangChain a menudo usa PINECONE_API_KEY y PINECONE_ENVIRONMENT directamente
    # PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Puede no ser necesario explícitamente
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag")
except ImportError:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    # PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "manuales-banco-rag")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("Error: Faltan claves API (Google o Pinecone).")
    st.stop()

# --- Modelos (LLM y Embeddings) ---
@st.cache_resource # Cachear recursos para no recargarlos
def load_models():
    # Usamos el embedder compatible con la ingesta (384 dim)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Usamos Gemini Pro
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", # Nombre simple funciona mejor
        temperature=0.1, # Un poco más creativo que 0
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )
    return llm, embeddings

llm, embeddings = load_models()

# --- Conexión a Pinecone (Base de Datos Vectorial Existente) ---
@st.cache_resource # Cachear la conexión al vector store
def load_vector_store():
    with st.spinner("Conectando a la base de conocimiento..."):
        # Inicializa Pinecone (puede requerir ajuste según versión instalada)
        # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) # Sintaxis antigua
        # pinecone.Pinecone(api_key=PINECONE_API_KEY) # Sintaxis más nueva

        # Conecta LangChain a tu índice Pinecone existente
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings # Usa el mismo modelo de embeddings que en la ingesta
        )
        retriever = vectorstore.as_retriever() # Obtiene el retriever directamente
    return retriever

retriever = load_vector_store()

# --- Lógica de Consulta RAG con LangChain ---
def generarConsulta(query, llm, retriever):
    system_prompt = (
        "Eres un agente de asistencia bancaria, amable y profesional. "
        "Tu única fuente de conocimiento son los manuales proporcionados del Banco Caroní. "
        "Usa el contexto recuperado para responder la pregunta. "
        "Si la respuesta no está en el contexto, indica amablemente que no tienes esa información. "
        "Mantén la respuesta concisa y clara. "
        "\n\n"
        "Contexto: {context}"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # Crea la cadena que primero obtiene documentos y luego responde
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invocamos la cadena con el input del usuario
    response = chain.invoke({"input": query})
    return response # Devuelve el diccionario completo

# --- Interfaz de Chat Streamlit ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if user_prompt := st.chat_input("¿Qué deseas saber sobre los manuales del banco?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            try:
                # Llama a la función de LangChain
                rag_response = generarConsulta(user_prompt, llm, retriever)

                # Extrae solo la respuesta de texto
                answer = rag_response.get("answer", "No se pudo generar una respuesta.")
                st.markdown(answer)

                # Opcional: Mostrar contexto recuperado (útil para depurar)
                with st.expander("Ver contexto recuperado"):
                    context_docs = rag_response.get("context", [])
                    if context_docs:
                        for i, doc in enumerate(context_docs):
                            st.write(f"**Fragmento {i+1}:**")
                            st.write(doc.page_content)
                    else:
                        st.write("No se recuperó contexto.")

                # Guarda la respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Ocurrió un error al generar la respuesta: {e}")
                # Guarda el error en el historial para depuración
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})