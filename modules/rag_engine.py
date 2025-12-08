import streamlit as st
import time
# --- CRÍTICO: Importamos las excepciones de Google para activar el Fallback ---
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.config import CONFIG

# Cacheamos la carga de recursos para que no se ejecute en cada interacción
@st.cache_resource(show_spinner="Cargando Cerebro del Asistente (Con Respaldo)...")
def load_models_and_retriever():
    try:
        # --- 1. Cargar Embeddings (HuggingFace Local) ---
        # Mantenemos tu configuración de embeddings locales en CPU
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True} 
        )

        # --- 2. Configurar LLM Principal (Flash 2.5) ---
        # Este es el modelo preferido.
        llm_primary = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_retries=0, # Fallar rápido para activar backup
            api_key=CONFIG["GOOGLE_API_KEY"]
        )

        # --- 3. Configurar LLM de Respaldo (Flash Lite) ---
        # Este modelo usa una cuota distinta, ideal para cuando se agota la principal.
        llm_backup = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.1,
            max_retries=2, 
            api_key=CONFIG["GOOGLE_API_KEY"]
        )

        # --- 4. Crear Sistema Robusto (Fallback) ---
        # Si el primario falla por cuota (429), se usa el backup automáticamente.
        llm_robust = llm_primary.with_fallbacks(
            [llm_backup],
            exceptions_to_handle=(ResourceExhausted, ServiceUnavailable, GoogleAPIError)
        )
        
        # --- 5. Conectar a Pinecone ---
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=CONFIG["PINECONE_INDEX_NAME"],
            embedding=embeddings
        )
        
        # --- 6. Configurar Retriever (Visión Aumentada) ---
        # Mantenemos k=12 ya que los tokens no son tu problema principal
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 12} 
        )
        
        print("✅ Modelos cargados: HuggingFace + Sistema Fallback + Alta Visión (k=12).")
        return llm_robust, retriever, vectorstore

    except Exception as e:
        st.error(f"Error crítico cargando modelos: {str(e)}")
        return None, None, None

@st.cache_resource
def create_rag_chain(_llm, _retriever, system_mode="🧠 Híbrido", chat_mode="💬 Conversacional"):
    """
    Crea la cadena RAG ajustando el System Prompt y el Modo de Chat.
    
    Args:
        system_mode: "🧠 Híbrido" (IA + Manuales) o "📜 Estricto" (Solo Manuales).
        chat_mode: "💬 Conversacional" (Con Memoria, 2 créditos) o "⚡ Puntual" (Sin Memoria, 1 crédito).
    """
    if not _llm or not _retriever:
        return None

    # --- 1. CONFIGURACIÓN DEL RECUPERADOR (RETRIEVER) ---
    
    if "Conversacional" in chat_mode:
        # MODO CONVERSACIONAL (Gasto: 2 llamadas)
        # Usa un prompt intermedio para reescribir la pregunta según el historial.
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
        # Creamos el retriever inteligente que entiende "contexto"
        retriever_to_use = create_history_aware_retriever(_llm, _retriever, contextualize_q_prompt)
    else:
        # MODO PUNTUAL (Gasto: 1 llamada - AHORRO DE CUOTA)
        # Usamos el retriever directo. La pregunta va directo a la base de datos.
        # No entiende "contexto" previo (ej: "¿y cuánto cuesta?"), pero ahorra 50% de cuota.
        retriever_to_use = _retriever

    
    # --- 2. CONFIGURACIÓN DE PERSONALIDAD (SYSTEM PROMPT) ---
    
    if "Estricto" in system_mode:
        # --- MODO ESTRICTO (SOLO MANUALES) ---
        qa_system_prompt = (
            "Eres un Asistente de Cumplimiento Normativo del Banco Caroní. "
            "Tu función es responder preguntas basándote ÚNICAMENTE en los manuales proporcionados.\n\n"
            
            "🔴 REGLAS ESTRICTAS DE RESPUESTA:\n"
            "1. **FUENTE ÚNICA:** Usa SOLO la información contenida en el 'Contexto' abajo.\n"
            "2. **CERO ALUCINACIONES:** Si la respuesta NO está explícitamente en el contexto, DEBES responder textualmente: "
            "'Lo siento, no cuento con información específica sobre este tema en mis manuales oficiales registrados.'\n"
            "3. **PROHIBIDO:** No uses conocimiento general, no inventes pasos, no asumas procedimientos de otros bancos.\n"
            "4. **CITAS:** Si encuentras la respuesta, menciona que proviene de la normativa interna.\n\n"
            
            "Contexto Normativo Recuperado:\n{context}"
        )
    else:
        # --- MODO HÍBRIDO (ORIGINAL - IA + MANUALES) ---
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
    
    # Devolvemos la cadena con el retriever seleccionado (Directo o Inteligente)
    return create_retrieval_chain(retriever_to_use, question_answer_chain)