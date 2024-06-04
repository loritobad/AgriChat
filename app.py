import os

from dotenv import load_dotenv


from BD.db_pinecone import conf_pinecone
from load_data.create_docs import create_documents, query_engine_open_ai
from promp.format_prompt import prompt_openai
from pdf2imgC.pdf import convert_pdf_to_img,clear_directory
from DETR.detect_table import  procesing_detect_table




import base64
import streamlit as st
import pandas as pd
from streamlit_chat import message
from streamlit.components.v1 import html

import logging
import sys
import shutil

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import streamlit as st
import os
import base64
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    pc = conf_pinecone()

            
    
    st.set_page_config(
        page_title="AgriChat",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    
    # Definir función para mostrar el historial de mensajes
    def show_chat_history(messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Inicializar historial de chat y variables de sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "ai", "content": "¡Hola! Soy Agrichat. ¿En qué puedo ayudarte?"})
    
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None
    
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
        

    upper_row = st.columns([0.2, 0.8])

    # Content in the upper row
    with upper_row[0]:
        api_key_openai = st.text_input("Introduce el API KEY de OpenAi", "")
        os.environ["OPENAI_API_KEY"] = api_key_openai

    with upper_row[1]:
        st.page_link("app.py", label="Chat", icon="🏠")
        st.page_link("pages/tables.py", label="Tabla detectadas", icon="1️⃣")
        st.page_link("pages/evaluation.py", label="Evaluacion", icon="1️⃣")

    # Define layout and containers
    HEIGHT = 400

    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        visor_pdf = st.container()
        
    with col2:
        barra_navegacion = st.container(border=True)
        panel_chat = st.container(height=HEIGHT + 15, border=True)
        panel_entrada_chat = st.container(border=True)

    with barra_navegacion:
        file = st.file_uploader("Upload a PDF file",
                                type="pdf",
                                accept_multiple_files=True,
                                key=st.session_state["file_uploader_key"])
        if file:
            st.session_state["uploaded_files"] = file
        
        if st.button("Borrar Archivos"):
            procesing_detect_table()
            st.session_state["file_uploader_key"] += 1
            st.session_state.messages = []
            st.session_state.pdf_bytes = None
            st.session_state.pdf_name = None
            st.rerun()
            

        

    ## Add contents
    pdf_placeholder = visor_pdf.empty()
    
    if file and file[0] is not None:
        pdf_directory = 'data_pdf'
        clear_directory()
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, filename)
                os.remove(file_path)

        for index, pdf in enumerate(file):  # Iterar sobre todos los archivos PDF cargados
            pdf_bytes = pdf.read()
            pdf_name = pdf.name 
            with open(os.path.join(pdf_directory, pdf_name), "wb") as f:
                f.write(pdf_bytes)

            # Guardar PDF en la sesión
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.pdf_name = pdf_name
            
            
            # Convertir el archivo en base64
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            # Insertar el PDF en HTML
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf">'
            pdf_placeholder.markdown(pdf_display, unsafe_allow_html=True)
            
            
            # Solo ejecutar si no hay PDF en la sesión
            if not st.session_state.get("pdf_processed", False):
                # Convertir el PDF a imágenes
                convert_pdf_to_img(pdf_bytes, pdf_name)
                create_documents(pc) #crea los embeddings en pinecone
                procesing_detect_table() # explota el modelo Table Transformer, probar asincronia
                
                st.session_state.pdf_processed = True  

            

            
        
    
        
             
        
        
    elif st.session_state.pdf_bytes:
        # Mostrar el PDF de la sesión si existe
        base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf">'
        pdf_placeholder.markdown(pdf_display, unsafe_allow_html=True)
    else:
        pdf_placeholder.write("Cargue documentos PDF")

    with panel_chat:
        st.title("Chat")
        show_chat_history(st.session_state.messages)

    with panel_entrada_chat: # Entrada de datos de chat
        if query_user := st.chat_input("¿Qué necesitas saber?"):
            context_augmented = query_engine_open_ai(pc, query_user)

            # Agregar el mensaje del usuario al historial de chat
            with panel_chat:
                with st.chat_message("human"):
                    st.markdown(query_user)
            st.session_state.messages.append({"role": "human", "content": query_user})

            # Enviar al modelo de lenguaje
            response_llm = prompt_openai(pc, context_augmented, query_user)

            with panel_chat:
                with st.chat_message("ai"):
                    st.markdown(response_llm)
            # Agregar la respuesta del modelo de lenguaje al historial de chat
            st.session_state.messages.append({"role": "ai", "content": response_llm})