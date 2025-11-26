import streamlit as st
from modules.database import pinecone_index, supabase_admin

def reset_all():
    print("⚠️ INICIANDO RESET COMPLETO DEL SISTEMA RAG...")
    
    # 1. BORRAR VECTORES EN PINECONE
    try:
        print("1. Borrando vectores de Pinecone...")
        pinecone_index.delete(delete_all=True)
        print("✅ Pinecone limpio.")
    except Exception as e:
        print(f"❌ Error en Pinecone: {e}")

    # 2. BORRAR REGISTROS EN SUPABASE (Tabla 'manuales')
    # Esto es necesario para que la app no crea que los archivos ya están subidos
    try:
        print("2. Limpiando tabla 'manuales' en Supabase...")
        # Borramos todos los registros donde el vector_count sea mayor a -1 (o sea, todos)
        supabase_admin.table('manuales').delete().gt('vector_count', -1).execute()
        print("✅ Tabla 'manuales' limpia.")
    except Exception as e:
        print(f"❌ Error en Supabase: {e}")

    # 3. (OPCIONAL) Borrar archivos físicos del Storage
    # Si quieres borrar también los PDFs de la nube, descomenta esto:
    """
    try:
        print("3. Borrando archivos de Storage...")
        files = supabase_admin.storage.from_("manuales-pdf").list()
        if files:
            file_paths = [f['name'] for f in files] # Ajustar según estructura de carpetas
            supabase_admin.storage.from_("manuales-pdf").remove(file_paths)
        print("✅ Storage limpio.")
    except Exception as e:
        print(f"❌ Error en Storage: {e}")
    """

    print("\n✨ SISTEMA REINICIADO. AHORA PUEDES SUBIR LOS MANUALES CON EL NUEVO MODELO.")

if __name__ == "__main__":
    reset_all()