import streamlit as st
import os
import pinecone

# --- Importaciones de LangChain ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

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
    # 1. Embeddings (Debe coincidir con la ingesta)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. LLM (Usando el modelo que confirmaste que funciona)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Asegúrate de que este sea el nombre correcto
        temperature=0.1,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )
    
    # 3. Conexión a Pinecone (Retriever)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    # Aumentamos a 7 fragmentos para dar más contexto
    retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
    
    return llm, retriever

llm, retriever = load_models_and_retriever()

# --- Lógica de Consulta RAG con Memoria (MODIFICADO) ---
@st.cache_resource
def create_rag_chain():
    
    # --- 1. PROMPT PARA RE-ESCRIBIR LA PREGUNTA ---
    # (Este se mantiene igual, es para la fluidez)
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
    
    # --- 2. PROMPT PARA RESPONDER LA PREGUNTA (LÓGICA MEJORADA) ---
    # Este es el cerebro del asistente, define la jerarquía de conocimiento.
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

# --- Interfaz de Chat Streamlit ---
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

                # Expansor de contexto (para depuración)
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