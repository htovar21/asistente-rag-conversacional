import streamlit as st
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.config import CONFIG

# Cacheamos la carga de recursos para que no se ejecute en cada interacción
@st.cache_resource(show_spinner="Cargando Cerebro del Asistente...")
def load_models_and_retriever():
    try:
        # 1. Cargar Embeddings (Configurado explícitamente para CPU)
        # Esto evita que busque CUDA y falle en entornos cloud
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True} 
        )

        # 2. Configurar LLM (Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_retries=2,
            api_key=CONFIG["GOOGLE_API_KEY"]
        )
        
        # 3. Conectar a Pinecone
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=CONFIG["PINECONE_INDEX_NAME"],
            embedding=embeddings
        )
        
        # 4. Configurar Retriever
        # K=5 es un buen balance. search_type="mmr" añade diversidad si tienes muchos docs repetidos.
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5} 
        )
        
        print("✅ Modelos y Retriever cargados exitosamente.")
        return llm, retriever, vectorstore

    except Exception as e:
        st.error(f"Error crítico cargando modelos: {str(e)}")
        # Retornamos None para manejarlo en la UI si es necesario
        return None, None, None

@st.cache_resource
def create_rag_chain(_llm, _retriever):
    if not _llm or not _retriever:
        return None

    # Prompt Contextualizador (Historial)
    contextualize_q_system_prompt = (
        "Dada la siguiente conversación y la última pregunta del usuario, "
        "reformula la última pregunta para que sea una **consulta de búsqueda independiente**. "
        "NO respondas la pregunta, solo reformúlala."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt),
         MessagesPlaceholder(variable_name="chat_history"),
         ("human", "{input}")]
    )
    
    # Creamos el retriever consciente del historial
    history_aware_retriever = create_history_aware_retriever(_llm, _retriever, contextualize_q_prompt)
    
    # Prompt de Respuesta (Experto Banco Caroní)
    qa_system_prompt = (
        "Eres un 'Asistente Experto' del Centro de Servicio al Usuario (CSU) del Banco Caroní. Eres amable, eficiente y tu objetivo es dar la mejor solución posible, combinando dos fuentes: (1) El 'Contexto' (manuales internos) y (2) Tu conocimiento general como IA experta en tecnología (Gemini)."

        "Tu regla de oro es la JERARQUÍA DE CONOCIMIENTO:"

        "1. PRIORIDAD MÁXIMA (Respuesta Basada en Manuales):"
        "   - SIEMPRE revisa el Contexto primero."
        "   - Si el Contexto (manuales) contiene la respuesta directa a la pregunta técnica del usuario (pasos, errores, soluciones), DEBES usar esa información como la fuente principal de tu respuesta."
        "   - Puedes usar tu conocimiento general (Gemini) para complementar o explicar de manera más sencilla el contexto, pero la solución principal debe venir del manual."
        "   - Al responder, indica que la información proviene de los manuales. (Ej: 'Según la base de conocimiento, los pasos son...') "

        "2. PRIORIDAD SECUNDARIA (Respuesta Basada en Conocimiento General):"
        "   - Si el Contexto está vacío O no es relevante para la pregunta técnica del usuario (ej. 'cómo desinstalar una impresora', 'pasos en Windows 7'):"
        "   - NO DIGAS 'No encontré la información'."
        "   - DEBES usar tu conocimiento general (Gemini) para proporcionar la mejor solución, los pasos o la explicación posible, como un experto en TI."
        "   - OBLIGATORIO: Después de dar tu respuesta basada en conocimiento general, DEBES AÑADIR la siguiente frase exacta: 'He identificado poco contenido sobre este tema en mi base de conocimiento. Te recomiendo que cuando se consiga la solución (si es un procedimiento interno), la subas a mi sistema usando la barra lateral para optimizar mi servicio.'"

        "3. EXCEPCIÓN DE SEGURIDAD (Sistemas Internos del Banco):"
        "   - Si la pregunta es sobre un procedimiento interno MUY específico del Banco Caroní (ej. 'Error 505 en Sistema IBS', 'clave del servidor X') Y el Contexto está vacío:"
        "   - NO INVENTES PASOS."
        "   - En este caso, responde: 'No encontré información específica sobre [tema] en la base de conocimiento. Como se trata de un sistema interno del banco, te recomiendo que cuando se consiga la solución, la subas a mi sistema usando la barra lateral para optimizar mi servicio.'"

        "4. SALUDOS Y CHARLA GENERAL:"
        "   - Responde amablemente usando tu conocimiento general."

        "Sé fluido y conversacional, usando el historial de chat para entender la conversación."
        "En resumen: Tu objetivo es solucionar el problema. Prioriza los manuales. Si no existen, usa tu cerebro (Gemini). Y si usas tu cerebro, pide al usuario que alimente la base de conocimiento."
        "\n\n"
        "Contexto: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt),
         MessagesPlaceholder(variable_name="chat_history"),
         ("human", "{input}")]
    )
    
    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)