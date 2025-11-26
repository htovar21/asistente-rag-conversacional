import json
import streamlit as st
from modules.database import supabase_admin

def get_user_sessions(username):
    """Obtiene la lista de chats del usuario ordenados por fecha."""
    try:
        response = supabase_admin.table('chat_sessions')\
            .select('*')\
            .eq('username', username)\
            .order('created_at', desc=True)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Error cargando sesiones: {e}")
        return []

def create_new_session(username, first_message_content):
    """Crea una nueva sesión usando el primer mensaje como título."""
    # Cortar título si es muy largo
    title = first_message_content[:30] + "..." if len(first_message_content) > 30 else first_message_content
    
    try:
        response = supabase_admin.table('chat_sessions').insert({
            'username': username,
            'title': title
        }).execute()
        
        # Retornar el ID de la nueva sesión
        if response.data:
            return response.data[0]['session_id']
        return None
    except Exception as e:
        st.error(f"Error creando sesión: {e}")
        return None

def load_chat_history(session_id):
    """Carga los mensajes de una sesión específica."""
    try:
        response = supabase_admin.table('chat_history')\
            .select('message_data')\
            .eq('session_id', session_id)\
            .order('created_at', desc=False)\
            .execute()
        
        messages = []
        for row in response.data:
            messages.append(json.loads(row['message_data']))
        return messages
    except Exception as e:
        st.error(f"Error cargando historial: {e}")
        return []

def save_message(username, session_id, role, content):
    """Guarda un mensaje en la BD vinculado a la sesión."""
    message_data = {"role": role, "content": content}
    try:
        supabase_admin.table('chat_history').insert({
            'username': username,
            'session_id': session_id,
            'message_data': json.dumps(message_data)
        }).execute()
    except Exception as e:
        st.error(f"Error guardando mensaje: {e}")

def delete_session(session_id):
    """Borra un chat entero (la cascada en SQL borrará los mensajes)."""
    try:
        supabase_admin.table('chat_sessions').delete().eq('session_id', session_id).execute()
    except Exception as e:
        st.error(f"Error borrando sesión: {e}")