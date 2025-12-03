import streamlit as st
import pandas as pd
from modules.database import supabase_admin
from modules.uploader import render_upload_section

@st.dialog("📚 Biblioteca de Documentos", width="large")
def open_library_modal(username, vector_store):
    
    tab_lista, tab_subir = st.tabs(["📂 Archivos Disponibles", "⬆️ Subir Nuevo"])

    # --- PESTAÑA 1: LISTADO Y DESCARGA ---
    with tab_lista:
        try:
            res = supabase_admin.table('manuales')\
                .select('filename, file_size, uploader_username, created_at, storage_path')\
                .order('created_at', desc=True).execute()
            
            if res.data:
                df = pd.DataFrame(res.data)
                
                if 'file_size' in df.columns:
                    df['Tamaño'] = df['file_size'].apply(lambda x: f"{x/(1024*1024):.2f} MB" if x else "0 MB")
                else:
                    df['Tamaño'] = "N/A"

                # --- CORRECCIÓN DE HORA (VENEZUELA) ---
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    # Convertir a Venezuela
                    df['created_at'] = df['created_at'].dt.tz_convert('America/Caracas')
                    # Formato 12H
                    df['Fecha'] = df['created_at'].dt.strftime('%d/%m/%Y %I:%M %p')
                # --------------------------------------

                view_df = df[['filename', 'Fecha', 'Tamaño', 'uploader_username']].rename(columns={
                    'filename': 'Documento',
                    'uploader_username': 'Autor',
                    'Fecha': 'Fecha Subida'
                })

                # Buscador
                search_lib = st.text_input("🔍 Buscar documento:", placeholder="Escribe para filtrar...")
                if search_lib:
                    view_df = view_df[view_df['Documento'].str.contains(search_lib, case=False, na=False)]

                st.dataframe(
                    view_df, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Documento": st.column_config.TextColumn("Documento", width="large"),
                        "Fecha Subida": st.column_config.TextColumn("Fecha Subida", width="medium")
                    }
                )
                
                st.divider()
                
                # Zona de Descarga
                c1, c2 = st.columns([0.7, 0.3])
                with c1:
                    file_to_download = st.selectbox(
                        "Selecciona archivo para descargar:", 
                        df['filename'].tolist(),
                        index=None,
                        placeholder="Elige un documento..."
                    )
                with c2:
                    st.write(""); st.write("")
                    if st.button("📥 Descargar", use_container_width=True, disabled=(file_to_download is None)):
                        path = df[df['filename'] == file_to_download]['storage_path'].iloc[0]
                        try:
                            data = supabase_admin.storage.from_("manuales-pdf").download(path)
                            st.download_button("💾 Guardar PDF", data, file_name=file_to_download, mime="application/pdf", type="primary")
                        except Exception as e: st.error("Error al descargar.")
            else:
                st.info("La biblioteca está vacía.")
        except Exception as e: st.error(f"Error: {e}")

    # --- PESTAÑA 2: SUBIR ---
    with tab_subir:
        st.info("Los archivos subidos aquí estarán disponibles para el Agente IA.")
        render_upload_section(username, vector_store, key_suffix="_modal")