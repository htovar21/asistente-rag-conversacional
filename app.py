import streamlit as st
import os
import pinecone
from pinecone import Pinecone # <--- Importación de Clase Pinecone
import re       
import hashlib  
import io       
from fpdf import FPDF
from pypdf import PdfReader
import json

# --- NUEVO: Importaciones de Supabase y Autenticador ---
from supabase import create_client, Client
import streamlit_authenticator as stauth
import bcrypt # Usaremos bcrypt directamente para el hash

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

# --- Configuración Inicial ---
st.set_page_config(
    page_title="Agente RAG Bancario (LangChain)",
    page_icon="🏦",
    layout="centered"
)

# --- Carga de Secretos y Conexión a Supabase ---
try:
    # Para pruebas locales
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "manuales-banco-rag")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Clave Anon/Public
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # Clave Secreta de Servicio
    COOKIE_SECRET_KEY = os.getenv("COOKIE_SECRET_KEY", "default_secret_key_123")
    
except ImportError:
    # Para despliegue en Streamlit Cloud
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "manuales-banco-rag")
    PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT")
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"] # Clave Anon/Public
    SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"] # Clave Secreta de Servicio
    COOKIE_SECRET_KEY = st.secrets.get("COOKIE_SECRET_KEY", "default_secret_key_123")

# Validación de Secretos
if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY, COOKIE_SECRET_KEY]):
    st.error("Error: Faltan claves API/Secretos (Google, Pinecone o Supabase).")
    st.stop()

# --- Conexión a Clientes ---
try:
    # Cliente ANÓNIMO (Solo para leer tabla de usuarios para el login)  
    supabase_anon: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Cliente ADMIN (Para todas las operaciones de escritura/lectura post-login)
    supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    # Conectar a Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Error fatal al conectar con los servicios: {e}")
    st.stop()


# --- LÓGICA DE AUTENTICACIÓN ---

@st.cache_data(ttl=600) # Cachear por 10 minutos
def fetch_users_from_db():
    try:
        # Usamos el cliente ANÓNIMO para leer la tabla de usuarios
        response = supabase_anon.table('usuarios').select('*').execute()
        users_list = response.data
        
        credentials = {'usernames': {}}
        for user in users_list:
            credentials['usernames'][user['username']] = {
                'name': user['nombre_completo'],
                'password': user['password_hash'], # Clave ya hasheada
                'role': user['role']
            }
        return credentials
    except Exception as e:
        st.error(f"Error conectando a la base de datos de usuarios: {e}")
        st.info("Asegúrate de que la tabla 'usuarios' tenga RLS activado y una política 'SELECT' para el rol 'public' (o 'anon').")
        return {'usernames': {}}

credentials = fetch_users_from_db()

authenticator = stauth.Authenticate(
    credentials,
    "csu_cookie_name",
    COOKIE_SECRET_KEY, # Clave secreta
    cookie_expiry_days=30
)

# --- RENDERIZAR EL LOGIN (VERSIÓN CORREGIDA) ---
authenticator.login()

authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if authentication_status == False:
    st.error('Usuario o contraseña incorrecto.')
elif authentication_status == None:
    st.warning('Por favor, ingrese su usuario y contraseña para continuar.')

# -----------------------------------------------
# --- APLICACIÓN PRINCIPAL (SI EL LOGIN ES EXITOSO) ---
# -----------------------------------------------
elif authentication_status == True:

    # Obtener el ROL del usuario logueado
    user_role = credentials['usernames'][username]['role']

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
        return llm, retriever, vectorstore

    llm, retriever, vector_store = load_models_and_retriever()

    # --- Lógica RAG (Prompt Experto) ---
    @st.cache_resource
    def create_rag_chain():
        contextualize_q_system_prompt = (
            "Dada la siguiente conversación y la última pregunta del usuario, "
            "reformula la última pregunta para que sea una **consulta de búsqueda independiente** "
            "que pueda entenderse sin el historial de chat. NO respondas la pregunta, "
            "solo reformúlala."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt),
             MessagesPlaceholder(variable_name="chat_history"),
             ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
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
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    rag_chain = create_rag_chain()

    # --- Función para limpiar IDs (Sin cambios) ---
    def sanitize_filename_to_ascii(filename):
        replacements = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', ' ': '_'}
        for char, replacement in replacements.items():
            filename = filename.replace(char, replacement)
        filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
        return filename

    # --- BARRA LATERAL (Sidebar) ---
    st.sidebar.title(f'Bienvenido *{name}*')
    st.sidebar.caption(f"Rol: {user_role.upper()}")
    authenticator.logout('Cerrar Sesión', 'sidebar')

    #Boton para limpiar historial de chat
    if st.sidebar.button("🧹 Limpiar Chat Actual"):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.divider()

    # --- SECCIÓN: AÑADIR NOTA (Visible para todos) ---
    st.sidebar.header("Añadir Nota Rápida (CSU)")
    if "pdf_to_download" in st.session_state:
        del st.session_state["pdf_to_download"]
    
    note_title = st.sidebar.text_input("Título de la Nota:", placeholder="Ej: Solución Error 503 Sistema X")
    note_text = st.sidebar.text_area("Escribe la nota o solución:", height=100, placeholder="Ej: El usuario debe borrar la caché...")

    if st.sidebar.button("Subir Nota"):
        if not note_title.strip() or not note_text.strip():
            st.sidebar.error("El Título y la Nota no pueden estar vacíos.")
        else:
            with st.spinner("Procesando y subiendo nota..."):
                try:
                    source_name = note_title
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
                    st.sidebar.success(f"¡Nota '{source_name}' subida con éxito!")

                    # Generar PDF para descargar
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 16) 
                    pdf.multi_cell(0, 10, source_name.encode('latin-1', 'replace').decode('latin-1'))
                    pdf.ln(5)
                    pdf.set_font("Helvetica", size=12) 
                    pdf.multi_cell(0, 10, note_text.encode('latin-1', 'replace').decode('latin-1'))
                    pdf_output = bytes(pdf.output()) 
                    
                    st.session_state["pdf_to_download"] = pdf_output
                    st.session_state["pdf_filename"] = f"{sanitize_filename_to_ascii(source_name)}.pdf" 

                except Exception as e:
                    st.sidebar.error(f"Error al subir nota: {e}")

    if "pdf_to_download" in st.session_state:
        st.sidebar.download_button(
            label="Descargar PDF de la Nota",
            data=st.session_state["pdf_to_download"],
            file_name=st.session_state["pdf_filename"],
            mime="application/pdf",
        )

    # --- SECCIÓN: SUBIR PDF (Visible para todos) ---
    st.sidebar.divider()
    st.sidebar.header("Subir Manual PDF")
    uploaded_file = st.sidebar.file_uploader("Selecciona un PDF:", type="pdf")
    
    if st.sidebar.button("Procesar y Subir PDF"):
        if uploaded_file:
            with st.spinner(f"Procesando '{uploaded_file.name}'..."):
                try:
                    filename_ascii = sanitize_filename_to_ascii(uploaded_file.name)
                    storage_path = f"{username}/{filename_ascii}"

                    # 1. Subir el archivo físico a Supabase Storage (Esto se hace primero)
                    bytes_data = bytes(uploaded_file.getvalue())
                    supabase_admin.storage.from_("manuales-pdf").upload(
                        path=storage_path,
                        file=bytes_data,
                        file_options={"cache-control": "3600", "upsert": "true"}
                    )
                    
                    # 2. Extraer Texto y Procesar (Para obtener el conteo)
                    pdf_stream = io.BytesIO(bytes_data)
                    reader = PdfReader(pdf_stream)
                    pdf_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                    
                    if pdf_text.strip():
                        # Borrar vectores viejos en Pinecone (Limpieza)
                        try:
                            pinecone_index.delete(filter={"source": uploaded_file.name})
                        except Exception:
                            pass # Si no existía, no pasa nada

                        # Segmentar el texto
                        new_doc = Document(page_content=pdf_text, metadata={"source": uploaded_file.name})
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
                        docs = text_splitter.split_documents([new_doc])
                        
                        # Generar IDs
                        ids = []
                        for i, doc in enumerate(docs):
                            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                            chunk_id = f"{filename_ascii}_{content_hash}_{i}"
                            ids.append(chunk_id)
                        
                        # 3. Subir vectores a Pinecone
                        vector_store.add_documents(docs, ids=ids)
                        
                        # 4. ¡AHORA SÍ! Guardar en Base de Datos con el CONTEO CORRECTO
                        # Lo hacemos aquí porque ahora sí sabemos que 'len(docs)' es la cantidad real
                        supabase_admin.table('manuales').upsert({
                            'filename': uploaded_file.name,
                            'storage_path': storage_path,
                            'uploader_username': username,
                            'vector_count': len(docs)  # <--- Aquí guardamos el número real
                        }, on_conflict='storage_path').execute() 
                        
                        st.sidebar.success(f"¡Manual '{uploaded_file.name}' subido! ({len(docs)} vectores).")
                    else:
                        st.sidebar.error("El PDF está vacío o es una imagen escaneada.")
                        
                except Exception as e:
                    st.sidebar.error(f"Error al subir PDF: {e}")
        else:
            st.sidebar.error("Por favor, selecciona un archivo PDF primero.")
        
    
    

    # --- PANEL DE ADMINISTRADOR (Visible solo para 'admin') ---
    if user_role == 'admin':
        st.sidebar.divider()
        st.sidebar.header("Panel de Administrador")
        
        # --- A. Gestión de Usuarios ---
        with st.sidebar.expander("Gestionar Usuarios (Admin)"):
            
            with st.form("Crear Usuario"):
                st.subheader("Crear Nuevo Usuario")
                new_username = st.text_input("Usuario (Username)")
                new_fullname = st.text_input("Nombre Completo")
                new_pass = st.text_input("Contraseña Temporal", type="password")
                new_role = st.selectbox("Rol", ["user", "admin"])
                
                if st.form_submit_button("Crear Usuario"):
                    if not all([new_username, new_fullname, new_pass]):
                        st.error("Por favor, llena todos los campos.")
                    else:
                        try:
                            # --- ¡CORRECCIÓN DE HASH USANDO BCRYPT! ---
                            password_bytes = new_pass.encode('utf-8')
                            salt = bcrypt.gensalt()
                            hashed_pass_bytes = bcrypt.hashpw(password_bytes, salt)
                            hashed_pass = hashed_pass_bytes.decode('utf-8')
                            
                            # Usar el cliente ADMIN
                            supabase_admin.table('usuarios').insert({
                                'username': new_username, 
                                'nombre_completo': new_fullname,
                                'password_hash': hashed_pass, 
                                'role': new_role
                            }).execute()
                            st.success(f"Usuario {new_username} creado.")
                            st.cache_data.clear() # Limpiar caché para recargar usuarios
                        except Exception as e:
                            st.error(f"Error al crear: {e}")
            
            st.divider()
            with st.form("Eliminar Usuario"):
                st.subheader("Eliminar Usuario")
                users_list = [u for u in credentials['usernames'].keys() if u != username]
                user_to_delete = st.selectbox("Selecciona usuario a eliminar", users_list)
                
                if st.form_submit_button("Eliminar Usuario", type="primary"):
                    try:
                        # Usar el cliente ADMIN
                        supabase_admin.table('usuarios').delete().eq('username', user_to_delete).execute()
                        st.warning(f"Usuario {user_to_delete} eliminado.")
                        st.cache_data.clear() # Limpiar caché
                    except Exception as e:
                        st.error(f"Error al eliminar: {e}")

        # --- B. Gestión de Base de Conocimiento (Admin) ---
        with st.sidebar.expander("Gestionar Base de Conocimiento (Admin)"):
            st.subheader("Eliminar Manuales")
            st.warning("PRECAUCIÓN: Esto elimina el manual de Pinecone y Supabase.")
            
            try:
                # Usar el cliente ADMIN
                manuales_resp = supabase_admin.table('manuales').select('filename, storage_path').execute()
                manuales_data = manuales_resp.data
                manuales_dict = {m['filename']: m['storage_path'] for m in manuales_data}
                
                if not manuales_dict:
                    st.info("No hay manuales en la base de datos para eliminar.")
                else:
                    manual_to_delete_name = st.selectbox("Selecciona manual a eliminar", manuales_dict.keys())
                    
                    if st.button("Eliminar Manual Permanentemente", type="primary"):
                        with st.spinner(f"Eliminando '{manual_to_delete_name}'..."):
                            
                            # 1. Eliminar de Pinecone (filtrando por 'source')
                            pinecone_index.delete(filter={"source": manual_to_delete_name})
                            
                            # 2. Eliminar de Supabase SQL (Usando el cliente ADMIN)
                            supabase_admin.table('manuales').delete().eq('filename', manual_to_delete_name).execute()
                            
                            # 3. Eliminar de Supabase Storage (Usando el cliente ADMIN)
                            storage_path = manuales_dict[manual_to_delete_name]
                            supabase_admin.storage.from_("manuales-pdf").remove([storage_path])
                            
                            st.success(f"Manual '{manual_to_delete_name}' eliminado.")
                            st.rerun() # Recargar la app para refrescar la lista
            
            except Exception as e:
                st.error(f"Error al cargar manuales: {e}")


    # --- INTERFAZ DE CHAT (Visible para todos los logueados) ---
    st.title("🏦 Asistente de Apoyo Operacional CSU")
    st.caption("Impulsado por los manuales internos.")
    
    # --- Lógica de Historial de Chat Persistente (Req. 5) ---
    if "messages" not in st.session_state or st.session_state.get("username") != username:
        st.session_state.messages = []
        # Cargar historial del usuario actual desde Supabase
        try:
            # Usar el cliente ADMIN
            response = supabase_admin.table('chat_history').select('message_data').eq('username', username).order('created_at', desc=False).execute()
            for row in response.data:
                # Convertir el string JSON de la BD de vuelta a un diccionario de Python
                st.session_state.messages.append(json.loads(row['message_data']))
        except Exception as e:
            st.error(f"Error al cargar el historial: {e}")
            
    st.session_state["username"] = username # Asegurar que el username esté en sesión

    # Muestra el historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de usuario
    if user_prompt := st.chat_input("¿En qué te puedo ayudar?"):
        
        # Guardar mensaje de usuario en DB y Sesión
        user_message_data = {"role": "user", "content": user_prompt}
        st.session_state.messages.append(user_message_data)
        with st.chat_message("user"):
            st.markdown(user_prompt)
        try:
            # Usar el cliente ADMIN
            supabase_admin.table('chat_history').insert({'username': username, 'message_data': json.dumps(user_message_data)}).execute()
        except Exception as e:
            st.error(f"Error al guardar mensaje: {e}")

        # Respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: # Historial (sin la pregunta actual)
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

                    # Guardar respuesta de asistente en DB y Sesión
                    assistant_message_data = {"role": "assistant", "content": answer}
                    st.session_state.messages.append(assistant_message_data)
                    try:
                        # Usar el cliente ADMIN
                        supabase_admin.table('chat_history').insert({'username': username, 'message_data': json.dumps(assistant_message_data)}).execute()
                    except Exception as e:
                        st.error(f"Error al guardar respuesta: {e}")
                        
                    with st.expander("Ver contexto recuperado (Depuración)", expanded=False):
                        context_docs = rag_response.get("context", [])
                        if context_docs:
                            for i, doc in enumerate(context_docs):
                                st.write(f"--- Fragmento {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}) ---")
                                st.write(doc.page_content)
                        else:
                            st.write("**No se recuperó ningún contexto de Pinecone.**") 

                except Exception as e:
                    error_msg = f"Ocurrió un error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})