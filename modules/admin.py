import streamlit as st
import bcrypt
from modules.database import supabase_admin, pinecone_index

def render_admin_panel(username, credentials):
    st.sidebar.divider()
    st.sidebar.header("Panel de Administrador")
    
    # A. Gestión de Usuarios
    with st.sidebar.expander("Gestionar Usuarios"):
        with st.form("Crear Usuario"):
            st.subheader("Nuevo Usuario")
            new_user = st.text_input("Username")
            new_name = st.text_input("Nombre")
            new_pass = st.text_input("Password", type="password")
            new_role = st.selectbox("Rol", ["user", "admin"])
            
            if st.form_submit_button("Crear"):
                if all([new_user, new_name, new_pass]):
                    try:
                        salt = bcrypt.gensalt()
                        hashed = bcrypt.hashpw(new_pass.encode(), salt).decode()
                        supabase_admin.table('usuarios').insert({
                            'username': new_user, 'nombre_completo': new_name,
                            'password_hash': hashed, 'role': new_role
                        }).execute()
                        st.success(f"Usuario {new_user} creado.")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Eliminar usuario
        users_list = [u for u in credentials['usernames'].keys() if u != username]
        user_del = st.selectbox("Borrar usuario", users_list)
        if st.button("Eliminar Usuario", type="primary"):
            supabase_admin.table('usuarios').delete().eq('username', user_del).execute()
            st.warning("Eliminado.")
            st.cache_data.clear()

    # B. Gestión Manuales
    with st.sidebar.expander("Gestionar Manuales"):
        manuales = supabase_admin.table('manuales').select('filename').execute().data
        manual_dict = {m['filename']: m['filename'] for m in manuales} # Simplificado
        
        if manual_dict:
            to_del = st.selectbox("Borrar Manual", manual_dict.keys())
            if st.button("Borrar Manualmente"):
                try:
                    pinecone_index.delete(filter={"source": to_del})
                    supabase_admin.table('manuales').delete().eq('filename', to_del).execute()
                    # Nota: Falta lógica de borrar de Storage para simplificar, agregar si es necesario
                    st.success("Manual eliminado.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))