import streamlit as st
import bcrypt
import pandas as pd
from modules.database import supabase_admin, pinecone_index

def render_admin_panel(username, credentials):
    st.sidebar.divider()
    st.sidebar.header("🛡️ Panel de Administrador")
    
    # --- A. VER LISTADO DE USUARIOS (MEJORADO CON BUSCADOR) ---
    with st.sidebar.expander("👥 Ver Usuarios Registrados", expanded=False):
        try:
            # 1. Consultar a Supabase
            response = supabase_admin.table('usuarios').select('role, username, nombre_completo').execute()
            users_data = response.data
            
            if users_data:
                # 2. Convertir a DataFrame de Pandas
                df = pd.DataFrame(users_data)
                
                # 3. Renombrar columnas para que se vea bien
                df = df.rename(columns={
                    "role": "Rol",
                    "username": "Usuario",
                    "nombre_completo": "Nombre"
                })
                
                # --- NUEVO: BUSCADOR MANUAL ---
                # Esto soluciona el problema de la lupa nativa en el sidebar
                search_term = st.text_input("🔍 Buscar usuario:", placeholder="Escribe nombre o usuario...")
                
                if search_term:
                    # Filtra si el término aparece en Usuario O en Nombre (sin importar mayúsculas)
                    mask = df.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)
                    df_filtered = df[mask]
                else:
                    df_filtered = df
                # ------------------------------

                # 4. Mostrar la tabla filtrada
                st.dataframe(
                    df_filtered, 
                    hide_index=True, 
                    use_container_width=True
                )
                st.caption(f"Total: {len(df_filtered)} usuarios.")
            else:
                st.info("No hay usuarios registrados.")
                
        except Exception as e:
            st.error(f"Error al cargar lista: {e}")

    # --- B. GESTIÓN DE USUARIOS (CREAR/BORRAR) ---
    with st.sidebar.expander("👤 Gestionar Cuentas"):
        # Pestañas para organizar mejor
        tab_crear, tab_borrar = st.tabs(["Crear", "Eliminar"])
        
        # 1. Crear Usuario
        with tab_crear:
            with st.form("Crear Usuario"):
                new_user = st.text_input("Usuario (Username)")
                new_name = st.text_input("Nombre Completo")
                new_pass = st.text_input("Contraseña", type="password")
                new_role = st.selectbox("Rol", ["user", "admin"])
                
                if st.form_submit_button("Crear Usuario"):
                    if all([new_user, new_name, new_pass]):
                        try:
                            # Hashear contraseña
                            salt = bcrypt.gensalt()
                            hashed = bcrypt.hashpw(new_pass.encode(), salt).decode()
                            
                            # Insertar en DB
                            supabase_admin.table('usuarios').insert({
                                'username': new_user, 
                                'nombre_completo': new_name,
                                'password_hash': hashed, 
                                'role': new_role
                            }).execute()
                            
                            st.success(f"Usuario '{new_user}' creado.")
                            st.cache_data.clear() # Limpiar caché para actualizar tabla
                            st.rerun() # Recargar
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Llena todos los campos.")

        # 2. Eliminar Usuario
        with tab_borrar:
            # Filtramos para no borrarse a sí mismo
            users_list = [u for u in credentials['usernames'].keys() if u != username]
            
            if users_list:
                user_del = st.selectbox("Selecciona usuario a eliminar", users_list)
                if st.button("🗑️ Eliminar Usuario", type="primary"):
                    try:
                        supabase_admin.table('usuarios').delete().eq('username', user_del).execute()
                        st.success(f"Usuario '{user_del}' eliminado.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("No hay otros usuarios para eliminar.")

    # --- C. GESTIÓN DE MANUALES ---
    with st.sidebar.expander("📚 Gestionar Base de Conocimiento"):
        st.caption("Borrado definitivo de manuales (Vector + Archivo + DB).")
        
        try:
            manuales = supabase_admin.table('manuales').select('filename').execute().data
            manual_dict = {m['filename']: m['filename'] for m in manuales}
            
            if manual_dict:
                to_del = st.selectbox("Selecciona Manual a Borrar", list(manual_dict.keys()))
                
                if st.button("🔥 Borrar Manual Definitivamente", type="primary"):
                    with st.spinner("Eliminando rastros..."):
                        try:
                            # 1. Borrar de Pinecone
                            pinecone_index.delete(filter={"source": to_del})
                            # 2. Borrar de SQL (Base de datos)
                            supabase_admin.table('manuales').delete().eq('filename', to_del).execute()
                            # 3. Nota: Para borrar de Storage requeriría la ruta exacta
                            
                            st.success("Manual eliminado del cerebro de la IA.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("No hay manuales cargados.")
                
        except Exception as e:
            st.error(f"Error cargando manuales: {e}")