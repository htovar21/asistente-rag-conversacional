import streamlit as st
import bcrypt
import pandas as pd
from modules.database import supabase_admin, pinecone_index

@st.dialog("Panel de Administración", width="large")
def open_admin_modal(username, credentials):
    
    # Pestañas limpias (sin emojis para look corporativo)
    tab_users, tab_docs = st.tabs(["Usuarios", "Base de Conocimiento"])

    # =======================================================
    # --- PESTAÑA 1: GESTIÓN DE USUARIOS ---
    # =======================================================
    with tab_users:
        try:
            res = supabase_admin.table('usuarios').select('username, nombre_completo, role, created_at').execute()
            if res.data:
                df_users = pd.DataFrame(res.data)
                
                # --- CORRECCIÓN DE HORA ---
                if 'created_at' in df_users.columns:
                    df_users['created_at'] = pd.to_datetime(df_users['created_at'])
                    try:
                        df_users['created_at'] = df_users['created_at'].dt.tz_convert('America/Caracas')
                    except: pass
                    df_users['Fecha Registro'] = df_users['created_at'].dt.strftime('%d/%m/%Y %I:%M %p')
                else:
                    df_users['Fecha Registro'] = "N/A"

                df_users = df_users.rename(columns={
                    "role": "Rol", "username": "Usuario", "nombre_completo": "Nombre"
                })
                
                # Input de búsqueda limpio (sin emoji)
                search_user = st.text_input("Buscar usuario:", placeholder="Nombre o usuario...")
                if search_user:
                    mask = df_users.apply(lambda x: x.astype(str).str.contains(search_user, case=False).any(), axis=1)
                    df_users = df_users[mask]

                st.dataframe(
                    df_users[['Usuario', 'Nombre', 'Rol', 'Fecha Registro']],
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Fecha Registro": st.column_config.TextColumn("Fecha Registro", width="medium"), 
                        "Nombre": st.column_config.TextColumn("Nombre", width="large")
                    }
                )
            else: 
                # Icono de información material
                st.info("No hay usuarios registrados.", icon=":material/group_off:")
        except Exception as e: 
            st.error(f"Error cargando usuarios: {e}", icon=":material/error:")
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Nuevo Usuario")
            with st.form("new_user_form", clear_on_submit=True):
                u = st.text_input("Usuario")
                n = st.text_input("Nombre Completo")
                p = st.text_input("Contraseña", type="password")
                r = st.selectbox("Rol", ["user", "admin"])
                
                # Botón con icono profesional
                if st.form_submit_button("Crear Usuario", icon=":material/person_add:"):
                    if u and p:
                        try:
                            s = bcrypt.gensalt()
                            h = bcrypt.hashpw(p.encode(), s).decode()
                            supabase_admin.table('usuarios').insert({
                                'username': u,
                                'nombre_completo': n,
                                'password_hash': h,
                                'role': r
                            }).execute()
                            st.success("Usuario creado exitosamente.", icon=":material/check_circle:")
                            st.rerun()
                        except Exception as e: st.error(f"Error: {e}", icon=":material/error:")
                    else:
                        st.warning("Usuario y contraseña son obligatorios.", icon=":material/warning:")

        with c2:
            st.markdown("#### Eliminar Usuario")
            users = [x for x in credentials['usernames'] if x != username]
            if users:
                d = st.selectbox("Seleccionar Usuario", users, index=None, placeholder="Elige usuario...")
                # Botón con icono profesional
                if st.button("Eliminar Usuario", disabled=(d is None), icon=":material/person_remove:"):
                    try:
                        supabase_admin.table('usuarios').delete().eq('username', d).execute()
                        st.success(f"Usuario '{d}' eliminado.", icon=":material/delete:")
                        st.rerun()
                    except Exception as e: st.error(f"Error: {e}", icon=":material/error:")
            else: 
                st.info("No hay otros usuarios para eliminar.", icon=":material/info:")

    # =======================================================
    # --- PESTAÑA 2: DOCUMENTOS ---
    # =======================================================
    with tab_docs:
        try:
            # --- CAMBIO IMPORTANTE: Consulta Relacional ---
            # Traemos todo de manuales Y el username de la tabla usuarios relacionada
            res = supabase_admin.table('manuales').select('*, usuarios(username)').order('created_at', desc=True).execute()
            
            if res.data:
                df = pd.DataFrame(res.data)
                
                # --- CAMBIO IMPORTANTE: Extracción de Autor ---
                # Si existe la relación, sacamos el nombre. Si es None (fue borrado), ponemos aviso.
                if 'usuarios' in df.columns:
                    df['uploader_username'] = df['usuarios'].apply(lambda x: x['username'] if x else "Usuario Eliminado")
                else:
                    df['uploader_username'] = "Desconocido"

                if 'file_size' in df.columns:
                    df['MB'] = df['file_size'].apply(lambda x: round(x/(1024*1024), 2) if x and x > 0 else 0)
                else: df['MB'] = 0

                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    try:
                        df['created_at'] = df['created_at'].dt.tz_convert('America/Caracas')
                    except: pass
                    df['Fecha'] = df['created_at'].dt.strftime('%d/%m/%Y %I:%M %p')
                else: df['Fecha'] = "Sin Fecha"

                view = df[['filename', 'Fecha', 'MB', 'uploader_username']].rename(columns={
                    'filename': 'Documento', 
                    'Fecha': 'Fecha Subida', 
                    'MB': 'Tamaño (MB)', 
                    'uploader_username': 'Autor'
                })
                
                # Input de búsqueda limpio (sin emoji)
                search_docs = st.text_input("Buscar manual:", placeholder="Nombre del archivo...")
                if search_docs:
                    view = view[view['Documento'].str.contains(search_docs, case=False, na=False)]

                st.dataframe(
                    view, hide_index=True, use_container_width=True,
                    column_config={
                        "Tamaño (MB)": st.column_config.NumberColumn(format="%.2f MB"), 
                        "Fecha Subida": st.column_config.TextColumn("Fecha Subida", width="medium"), 
                        "Documento": st.column_config.TextColumn("Documento", width="large")
                    }
                )
                
                st.divider()
                # Warning con icono Material
                st.warning("Zona de Eliminación Definitiva", icon=":material/warning:")
                
                c_sel, c_btn = st.columns([0.7, 0.3])
                with c_sel:
                    to_del = st.selectbox("Selecciona Manual:", df['filename'].tolist(), index=None, placeholder="Elige documento a borrar...")
                with c_btn:
                    st.write(""); st.write("")
                    # Botón de borrado crítico con icono profesional
                    if st.button("Borrar Manual", type="primary", use_container_width=True, disabled=(to_del is None), icon=":material/delete_forever:"):
                        with st.spinner("Eliminando registros..."):
                            try:
                                pinecone_index.delete(filter={"source": to_del})
                                supabase_admin.table('manuales').delete().eq('filename', to_del).execute()
                                st.success("Documento eliminado correctamente.", icon=":material/check_circle:")
                                st.rerun()
                            except Exception as e: st.error(str(e), icon=":material/error:")
            else: 
                st.info("La biblioteca está vacía.", icon=":material/folder_off:")
        except Exception as e: st.error(f"Error cargando documentos: {e}", icon=":material/error:")