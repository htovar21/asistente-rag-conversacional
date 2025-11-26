import streamlit as st
import streamlit_authenticator as stauth
from modules.database import supabase_anon
from modules.config import CONFIG

@st.cache_data(ttl=600)
def fetch_users_from_db():
    try:
        response = supabase_anon.table('usuarios').select('*').execute()
        users_list = response.data
        
        credentials = {'usernames': {}}
        for user in users_list:
            credentials['usernames'][user['username']] = {
                'name': user['nombre_completo'],
                'password': user['password_hash'],
                'role': user['role']
            }
        return credentials
    except Exception as e:
        st.error(f"Error DB Usuarios: {e}")
        return {'usernames': {}}

def run_login():
    credentials = fetch_users_from_db()
    
    authenticator = stauth.Authenticate(
        credentials,
        "csu_cookie_name",
        CONFIG["COOKIE_SECRET_KEY"],
        cookie_expiry_days=30
    )
    
    authenticator.login()
    
    return authenticator, st.session_state.get("authentication_status"), st.session_state.get("username"), credentials