import streamlit as st
import pandas as pd
import io
import zipfile
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
                
                # --- PROCESAMIENTO DE DATOS ---
                # Calcular MB
                if 'file_size' in df.columns:
                    df['Tamaño'] = df['file_size'].apply(lambda x: f"{x/(1024*1024):.2f} MB" if x else "0 MB")
                else:
                    df['Tamaño'] = "N/A"

                # Formatear Fecha (Con tu mejora de formato 12H)
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    try:
                        df['created_at'] = df['created_at'].dt.tz_convert('America/Caracas')
                    except: pass 
                    # Formato legible: DD/MM/YYYY 12:00 PM
                    df['Fecha'] = df['created_at'].dt.strftime('%d/%m/%Y %I:%M %p')

                # Preparar DataFrame Visual
                if "lib_select_all" not in st.session_state:
                    st.session_state.lib_select_all = False
                
                # --- SECCIÓN 1: TABLA Y DESCARGA MASIVA (ZIP) ---
                c_search, c_all, c_none = st.columns([0.6, 0.2, 0.2])
                with c_search:
                    search_lib = st.text_input("🔍 Buscar:", placeholder="Filtrar por nombre...", label_visibility="collapsed")
                with c_all:
                    if st.button("☑️ Todos", use_container_width=True, help="Marcar todos"):
                        st.session_state.lib_select_all = True
                        st.session_state.lib_editor_key = st.session_state.get("lib_editor_key", 0) + 1
                with c_none:
                    if st.button("⬜ Ninguno", use_container_width=True, help="Desmarcar todo"):
                        st.session_state.lib_select_all = False
                        st.session_state.lib_editor_key = st.session_state.get("lib_editor_key", 0) + 1
                        if "zip_data" in st.session_state: del st.session_state.zip_data

                # Preparamos los datos para la tabla
                view_df = df[['filename', 'Fecha', 'Tamaño', 'uploader_username', 'storage_path']].rename(columns={
                    'filename': 'Documento',
                    'uploader_username': 'Autor',
                    'Fecha': 'Fecha Subida'
                })
                
                # Filtrar si hay búsqueda
                if search_lib:
                    view_df = view_df[view_df['Documento'].str.contains(search_lib, case=False, na=False)]

                # Insertar columna de selección
                view_df.insert(0, "Seleccionar", st.session_state.lib_select_all)

                # Tabla Interactiva
                editor_key = f"editor_lib_{st.session_state.get('lib_editor_key', 0)}"
                
                edited_df = st.data_editor(
                    view_df,
                    hide_index=True,
                    use_container_width=True,
                    key=editor_key,
                    column_config={
                        "Seleccionar": st.column_config.CheckboxColumn("✔", width="small", default=False),
                        "Documento": st.column_config.TextColumn("Documento", width="large", disabled=True),
                        "Fecha Subida": st.column_config.TextColumn("Fecha Subida", width="medium", disabled=True),
                        "Tamaño": st.column_config.TextColumn("Tamaño", width="small", disabled=True),
                        "Autor": st.column_config.TextColumn("Autor", width="small", disabled=True),
                        "storage_path": None
                    }
                )

                # Lógica ZIP
                selected_rows = edited_df[edited_df["Seleccionar"] == True]
                count = len(selected_rows)
                
                st.caption(f"Seleccionados: {count}")
                
                if count > 0:
                    if st.button(f"📦 Generar ZIP ({count})", type="primary", use_container_width=True):
                        with st.spinner(f"Comprimiendo {count} archivos..."):
                            try:
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                    prog_bar = st.progress(0)
                                    for i, (_, row) in enumerate(selected_rows.iterrows()):
                                        file_content = supabase_admin.storage.from_("manuales-pdf").download(row['storage_path'])
                                        zf.writestr(row['Documento'], file_content)
                                        prog_bar.progress((i + 1) / count)
                                    prog_bar.empty()
                                
                                st.session_state.zip_data = zip_buffer.getvalue()
                                st.session_state.zip_name = f"manuales_seleccionados_{pd.Timestamp.now().strftime('%H%M%S')}.zip"
                            except Exception as e: st.error(f"Error generando ZIP: {e}")

                    if "zip_data" in st.session_state:
                        st.download_button(
                            label="💾 Guardar Archivo ZIP",
                            data=st.session_state.zip_data,
                            file_name=st.session_state.zip_name,
                            mime="application/zip",
                            type="secondary",
                            use_container_width=True
                        )
                
                st.divider()

                # --- SECCIÓN 2: DESCARGA INDIVIDUAL (NUEVA) ---
                st.subheader("📥 Descarga Individual")
                
                c_ind_sel, c_ind_btn = st.columns([0.7, 0.3])
                
                with c_ind_sel:
                    file_to_download = st.selectbox(
                        "Selecciona un archivo:", 
                        df['filename'].tolist(),
                        index=None,
                        placeholder="Buscar en la lista...",
                        label_visibility="collapsed",
                        key="sb_individual_download"
                    )

                with c_ind_btn:
                    # Lógica para preparar la descarga individual
                    if file_to_download:
                        path = df[df['filename'] == file_to_download]['storage_path'].iloc[0]
                        
                        # Usamos un botón intermedio para no saturar si el usuario cambia mucho el selectbox
                        if st.button("🔄 Preparar", use_container_width=True, key="btn_prep_single"):
                            try:
                                with st.spinner("Descargando..."):
                                    ind_data = supabase_admin.storage.from_("manuales-pdf").download(path)
                                    st.session_state.ind_file_data = ind_data
                                    st.session_state.ind_file_name = file_to_download
                                    
                            except Exception as e: st.error("Error al descargar.")
                
                # Mostrar botón de descarga real si los datos están listos y coinciden con la selección
                if "ind_file_data" in st.session_state and st.session_state.get("ind_file_name") == file_to_download:
                    st.download_button(
                        label=f"💾 Guardar {file_to_download}",
                        data=st.session_state.ind_file_data,
                        file_name=st.session_state.ind_file_name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )

            else:
                st.info("La biblioteca está vacía.")
        except Exception as e: st.error(f"Error: {e}")

    # --- PESTAÑA 2: SUBIR ---
    with tab_subir:
        st.info("Los archivos subidos aquí estarán disponibles para el Agente IA.")
        # Limpieza de memoria al cambiar de pestaña
        if "zip_data" in st.session_state: del st.session_state.zip_data
        if "ind_file_data" in st.session_state: del st.session_state.ind_file_data
        render_upload_section(username, vector_store, key_suffix="_modal", use_modal=False)