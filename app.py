import streamlit as st
import os
import pinecone
import re       # Para limpiar IDs
import hashlib  # Para crear IDs únicos
import io       # <--- AÑADIDO: Para manejar archivos en memoria

# --- Importaciones de LangChain ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- (NUEVO) Importar FPDF para crear el PDF ---
from fpdf import FPDF

# --- (NUEVO) Importar el Lector de PDF ---
from pypdf import PdfReader

# --- Configuración Inicial ---
st.set_page_config(
    page_title="Agente RAG Bancario (LangChain)",
    page_icon="🏦",
    layout="centered"
)
st.title("🏦 Asistente de Apoyo Operacional CSU")
st.caption("Impulsado por los manuales internos (Windows, Impresoras, Sistemas Bancarios).")

# --- Carga de Secretos y Configuración ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag")
except ImportError:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "manuales-banco-rag")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("Error: Faltan claves API (Google o Pinecone).")
    st.stop()

# --- Modelos (LLM y Embeddings) ---
@st.cache_resource
def load_models_and_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Usando el modelo que funcionó
        temperature=0.1,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
    
    # Devolvemos el vectorstore (necesario para añadir docs)
    return llm, retriever, vectorstore

llm, retriever, vector_store = load_models_and_retriever()


# --- Lógica de Consulta RAG con Memoria ---
@st.cache_resource
def create_rag_chain():
    
    # --- 1. PROMPT PARA RE-ESCRIBIR LA PREGUNTA ---
    contextualize_q_system_prompt = (
        "Dada la siguiente conversación y la última pregunta del usuario, "
        "reformula la última pregunta para que sea una **consulta de búsqueda independiente** "
        "que pueda entenderse sin el historial de chat. NO respondas la pregunta, "
        "solo reformúlala."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # --- 2. PROMPT PARA RESPONDER LA PREGUNTA (Asistente Experto) ---
    qa_system_prompt = (
        "Eres un 'Asistente Experto' del Centro de Servicio al Usuario (CSU) del Banco Caroní. Eres amable, eficiente y tu objetivo es dar soluciones operacionales claras."
        "Tienes dos fuentes de conocimiento: (1) El 'Contexto' (manuales internos de Windows, impresoras y sistemas) y (2) Tu conocimiento general (inherente de Gemini)."
        
        "**Tu regla de oro es priorizar los manuales:**"
        
        "1. **SI LA PREGUNTA ES TÉCNICA (pasos, soluciones, cómo hacer algo, errores):**"
        "   - **Primero,** busca la respuesta **ÚNICAMENTE** en el **Contexto** proporcionado."
        "   - **Si el Contexto contiene la respuesta:** Responde detalladamente basándote en él. Cita los pasos si es necesario."
        "   - **Si el Contexto está vacío O no es relevante:** DEBES decir: 'No encontré los pasos o la solución para [tema] en mi base de conocimiento de manuales.'"
        "   - **IMPORTANTE:** Nunca inventes procedimientos técnicos ni des pasos si no están en el Contexto."
        
        "2. **SI LA PREGUNTA ES GENERAL (definiciones, saludos, 'hola'):**"
        "   - **Si el Contexto está vacío:** Eres libre de usar tu **conocimiento general inherente** para responder (ej. 'Hola', o 'Una impresora es un dispositivo que...')."
        "   - **Si el Contexto SÍ tiene la definición:** Prefiere el Contexto, pero puedes complementarlo si es útil."
        
        "Sé fluido y conversacional, usando el historial de chat para entender la conversación."
        "\n\n"
        "Contexto: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

rag_chain = create_rag_chain()

# --- Función para limpiar IDs ---
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

# --- BARRA LATERAL (Sidebar) ---

# --- Sección: Subir Nota de Texto ---
st.sidebar.header("Añadir Nota Rápida (CSU)")
st.sidebar.caption("Añadir una solución o nota de texto a la base de conocimiento.")

if "pdf_to_download" in st.session_state:
    del st.session_state["pdf_to_download"]

note_text = st.sidebar.text_area("Escribe una nota o solución:", height=100,
                                 placeholder="Ej: 'Error 503 en Sistema X: Borrar caché del navegador.'")

if st.sidebar.button("Subir Nota"):
    if not note_text.strip():
        st.sidebar.error("La nota no puede estar vacía.")
    else:
        with st.spinner("Procesando y subiendo nota..."):
            try:
                # Lógica para subir la NOTA (como la tenías)
                source_name = "Nota_Manual_CSU" 
                new_doc = Document(page_content=note_text, metadata={"source": source_name})
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
                docs = text_splitter.split_documents([new_doc])
                
                ids = []
                for i, doc in enumerate(docs):
                    content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                    filename_ascii = sanitize_filename_to_ascii(source_name)
                    chunk_id = f"{filename_ascii}_{content_hash}_{i}" 
                    ids.append(chunk_id)

                vector_store.add_documents(docs, ids=ids)
                st.sidebar.success("¡Nota subida con éxito!")

                # Lógica para generar PDF y descargarlo (como la tenías)
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=12) 
                pdf.multi_cell(0, 10, note_text.encode('latin-1', 'replace').decode('latin-1'))
                pdf_output = bytes(pdf.output()) 
                st.session_state["pdf_to_download"] = pdf_output
                st.session_state["pdf_filename"] = f"Nota_CSU_{ids[0]}.pdf" 

            except Exception as e:
                st.sidebar.error(f"Error al subir nota: {e}")

if "pdf_to_download" in st.session_state:
    st.sidebar.download_button(
        label="Descargar PDF de la Nota Guardada",
        data=st.session_state["pdf_to_download"],
        file_name=st.session_state["pdf_filename"],
        mime="application/pdf",
    )

# --- (NUEVO) Sección: Subir Manual PDF ---
st.sidebar.divider() # Separador visual
st.sidebar.header("Subir Manual PDF (Admin)")
st.sidebar.caption("Subir un manual completo (.pdf) a la base de conocimiento.")

uploaded_file = st.sidebar.file_uploader(
    "Selecciona un archivo PDF:",
    type="pdf",
    accept_multiple_files=False # Permitir solo un archivo a la vez
)

if st.sidebar.button("Procesar y Subir PDF"):
    if uploaded_file is None:
        st.sidebar.error("Por favor, selecciona un archivo PDF primero.")
    else:
        with st.spinner(f"Procesando '{uploaded_file.name}'... (Esto puede tardar)"):
            try:
                # 1. Extraer Texto del PDF en memoria
                bytes_data = uploaded_file.getvalue()
                pdf_stream = io.BytesIO(bytes_data)
                reader = PdfReader(pdf_stream)
                
                pdf_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text
                
                if not pdf_text.strip():
                    st.sidebar.error("El PDF está vacío o no se pudo extraer texto.")
                else:
                    # 2. Convertir texto a Documento LangChain
                    source_name = uploaded_file.name
                    new_doc = Document(page_content=pdf_text, metadata={"source": source_name})
                    
                    # 3. Segmentar (Chunking)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=20
                    )
                    docs = text_splitter.split_documents([new_doc])
                    
                    # 4. Generar IDs (Misma lógica de ingesta - basada en chunk)
                    ids = []
                    for i, doc in enumerate(docs):
                        # Usamos el hash del contenido para evitar duplicados si el *chunk* es idéntico
                        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                        filename_ascii = sanitize_filename_to_ascii(source_name)
                        chunk_id = f"{filename_ascii}_{content_hash}_{i}" 
                        ids.append(chunk_id)
                    
                    # 5. Subir a Pinecone (Upsert)
                    vector_store.add_documents(docs, ids=ids)
                    
                    st.sidebar.success(f"¡Manual '{uploaded_file.name}' subido con éxito! ({len(docs)} fragmentos indexados)")
                
            except Exception as e:
                st.sidebar.error(f"Error al subir PDF: {e}")

# --- Interfaz de Chat Streamlit (SIN CAMBIOS) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("¿En qué te puedo ayudar?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            try:
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                rag_response = rag_chain.invoke({
                    "input": user_prompt,
                    "chat_history": chat_history
                })

                answer = rag_response.get("answer", "Error: No se generó respuesta.")
                st.markdown(answer)

                with st.expander("Ver contexto recuperado (Depuración)", expanded=False):
                    context_docs = rag_response.get("context", [])
                    if context_docs:
                        for i, doc in enumerate(context_docs):
                            st.write(f"--- Fragmento {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}) ---")
                            st.write(doc.page_content)
                    else:
                        st.write("**No se recuperó ningún contexto de Pinecone.**") 

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Ocurrió un error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})