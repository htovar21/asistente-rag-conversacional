import streamlit as st
import bcrypt
import pandas as pd
from modules.database import supabase_admin, pinecone_index

@st.dialog("🛡️ Panel de Administración", width="large")
def open_admin_modal(username, credentials):
    
    tab_users, tab_docs = st.tabs(["👥 Usuarios", "📚 Base de Conocimiento"])

    # --- PESTAÑA 1: GESTIÓN DE USUARIOS ---
    with tab_users:
        try:
            res = supabase_admin.table('usuarios').select('username, nombre_completo, role').execute()
            if res.data:
                df_users = pd.DataFrame(res.data)
                df_users = df_users.rename(columns={
                    "role": "Rol", 
                    "username": "Usuario", 
                    "nombre_completo": "Nombre"
                })
                
                # BUSCADOR USUARIOS
                search_user = st.text_input("🔍 Buscar usuario:", placeholder="Escribe nombre o usuario...")
                if search_user:
                    mask = df_users.apply(lambda x: x.astype(str).str.contains(search_user, case=False).any(), axis=1)
                    df_users = df_users[mask]

                st.dataframe(df_users, hide_index=True, use_container_width=True)
            else:
                st.info("No hay usuarios registrados.")
        except: st.error("Error cargando usuarios.")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Nuevo Usuario")
            with st.form("new_user_form", clear_on_submit=True):
                u = st.text_input("User"); n = st.text_input("Nombre"); p = st.text_input("Pass", type="password")
                r = st.selectbox("Rol", ["user", "admin"])
                if st.form_submit_button("Crear"):
                    if u and p:
                        try:
                            s = bcrypt.gensalt(); h = bcrypt.hashpw(p.encode(), s).decode()
                            supabase_admin.table('usuarios').insert({'username':u,'nombre_completo':n,'password_hash':h,'role':r}).execute()
                            st.success("Creado"); st.rerun()
                        except Exception as e: st.error(f"Error: {e}")
        with c2:
            st.markdown("#### Eliminar Usuario")
            users = [x for x in credentials['usernames'] if x != username]
            if users:
                d = st.selectbox("Usuario", users)
                if st.button("Eliminar"):
                    supabase_admin.table('usuarios').delete().eq('username', d).execute()
                    st.success("Eliminado"); st.rerun()
            else: st.info("Sin usuarios adicionales.")

    # --- PESTAÑA 2: DOCUMENTOS (CON BUSCADOR) ---
    with tab_docs:
        try:
            res = supabase_admin.table('manuales').select('*').order('created_at', desc=True).execute()
            if res.data:
                df = pd.DataFrame(res.data)
                
                # Calcular MB
                if 'file_size' in df.columns:
                    df['MB'] = df['file_size'].apply(lambda x: round(x/(1024*1024), 2) if x and x > 0 else 0)
                else: df['MB'] = 0

                # Formatear Fecha
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    df['Fecha'] = df['created_at'].dt.strftime('%d/%m/%Y %H:%M')
                else: df['Fecha'] = "Sin Fecha"

                # Vista Ordenada
                view = df[['filename', 'Fecha', 'MB', 'uploader_username']].rename(columns={
                    'filename': 'Documento', 
                    'Fecha': 'Fecha Subida',
                    'MB': 'Tamaño (MB)',
                    'uploader_username': 'Autor'
                })
                
                # --- BUSCADOR DE DOCUMENTOS ---
                search_docs = st.text_input("🔍 Buscar manual:", placeholder="Escribe el nombre del archivo...")
                if search_docs:
                    view = view[view['Documento'].str.contains(search_docs, case=False, na=False)]
                # ------------------------------

                st.dataframe(
                    view, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Tamaño (MB)": st.column_config.NumberColumn(format="%.2f MB"),
                        "Fecha Subida": st.column_config.TextColumn("Fecha Subida", width="medium"),
                        "Documento": st.column_config.TextColumn("Documento", width="large")
                    }
                )
                st.caption(f"Mostrando {len(view)} documentos.")
                
                st.divider()
                st.warning("⚠️ Zona de Peligro: Eliminación Definitiva")
                c_sel, c_btn = st.columns([0.7, 0.3])
                with c_sel:
                    # El selector usa la lista completa original para asegurar que puedas borrar aunque filtres mal
                    to_del = st.selectbox("Selecciona Manual:", df['filename'].tolist())
                with c_btn:
                    st.write(""); st.write("")
                    if st.button("🔥 Borrar", type="primary", use_container_width=True):
                        with st.spinner("Eliminando..."):
                            try:
                                pinecone_index.delete(filter={"source": to_del})
                                supabase_admin.table('manuales').delete().eq('filename', to_del).execute()
                                st.success("Eliminado."); st.rerun()
                            except Exception as e: st.error(str(e))
            else: st.info("La biblioteca está vacía.")
        except Exception as e: st.error(f"Error: {e}")

def render_admin_panel(username, credentials):
    if st.sidebar.button("⚙️ Panel Admin", type="primary", use_container_width=True):
        open_admin_modal(username, credentials)