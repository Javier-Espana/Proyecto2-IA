import streamlit as st
from vector_db import load_documents_to_pinecone, query_similar_docs
from utils import generate_response, validate_query, count_tokens
from config import AppConfig, APIConfig, config_errors
import os
from typing import List
from langchain.schema import Document

# ----------------------------
# Configuración inicial
# ----------------------------

# Verificar errores de configuración
if config_errors:
    st.error(f"Errores de configuración: {config_errors}")
    st.stop()

# Configurar página
st.set_page_config(
    page_title=AppConfig.UI_SETTINGS['page_title'],
    page_icon=AppConfig.UI_SETTINGS['page_icon'],
    layout=AppConfig.UI_SETTINGS['layout']
)

# ----------------------------
# Funciones auxiliares
# ----------------------------

def display_document_uploader() -> List[str]:
    """Componente para subir archivos y devolver rutas temporales"""
    uploaded_files = st.file_uploader(
        "📤 Sube documentos técnicos (PDF o TXT)",
        type=AppConfig.DOCUMENT_SETTINGS['allowed_extensions'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        temp_paths = []
        for file in uploaded_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(temp_path)
        return temp_paths
    return []

def display_answer(response: dict) -> None:
    """Muestra la respuesta formateada"""
    st.subheader("💡 Respuesta")
    st.markdown(response["answer"])
    
    with st.expander("🔍 Contexto utilizado"):
        st.write(response["context_used"])
        st.caption(f"Fuentes: {', '.join(response['sources'])}")

# ----------------------------
# Interfaz principal
# ----------------------------

def main():
    # Sidebar con información
    with st.sidebar:
        st.image("assets/logo.png", width=200)
        st.markdown(f"### {AppConfig.UI_SETTINGS['page_title']}")
        st.markdown("Asistente de consultas técnicas con IA")
        st.markdown("---")
        st.markdown("**Modelo:** " + APIConfig.OPENAI['model_name'])
        st.markdown("**Temperatura:** " + str(APIConfig.OPENAI['temperature']))
        st.markdown("---")
        
        if st.button("🧹 Limpiar historial"):
            st.session_state.messages = []
    
    # Título principal
    st.title(f"{AppConfig.UI_SETTINGS['page_icon']} {AppConfig.UI_SETTINGS['page_title']}")
    st.caption("Pregunta sobre cualquier tema técnico y el asistente buscará en la base de conocimiento")
    
    # Modo administrador (toggle)
    if st.checkbox("🔒 Mostrar opciones de administrador"):
        st.warning("Modo administrador activado")
        temp_paths = display_document_uploader()
        
        if temp_paths:
            with st.spinner(f"Cargando {len(temp_paths)} documentos..."):
                try:
                    load_documents_to_pinecone(
                        temp_paths,
                        namespace=AppConfig.DOCUMENT_SETTINGS['default_namespace']
                    )
                    st.success("Documentos cargados exitosamente!")
                except Exception as e:
                    st.error(f"Error al cargar documentos: {str(e)}")
            
            # Limpiar archivos temporales
            for path in temp_paths:
                os.remove(path)
    
    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input de usuario
    user_query = st.chat_input("Escribe tu pregunta técnica...")
    
    if user_query and validate_query(user_query):
        # Agregar pregunta al historial
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Buscar y generar respuesta
        with st.spinner("Buscando en la base de conocimiento..."):
            try:
                # 1. Búsqueda semántica
                similar_docs = query_similar_docs(
                    user_query,
                    k=AppConfig.SEARCH_SETTINGS['max_results'],
                    namespace=AppConfig.DOCUMENT_SETTINGS['default_namespace']
                )
                
                # 2. Generar respuesta
                response = generate_response(user_query, similar_docs)
                
                # Mostrar respuesta
                with st.chat_message("assistant"):
                    display_answer(response)
                
                # Guardar en historial
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
                
                # Métricas (debug)
                st.sidebar.caption(f"Tokens consulta: {count_tokens(user_query)}")
                st.sidebar.caption(f"Documentos encontrados: {len(similar_docs)}")
                
            except Exception as e:
                st.error(f"Error al procesar la consulta: {str(e)}")
    elif user_query:
        st.warning("La pregunta debe tener al menos 5 caracteres y menos de 300 tokens")

# ----------------------------
# Ejecución
# ----------------------------

if __name__ == "__main__":
    main()