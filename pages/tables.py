from streamlit_elements import elements, mui, html
import streamlit as st
import os
import pandas as pd
import csv


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



col3, col4 = st.columns([0.2, 0.8])





with col3:
   
    st.page_link("app.py", label="Chat", icon="🏠")
    st.page_link("pages/tables.py", label="Tabla detectadas", icon="1️⃣")
    st.page_link("pages/evaluation.py", label="Evaluacion", icon="1️⃣")

with col4:
    tablas =  st.container(height=700, border=True)
   
   
     

with tablas:
    csv_directory = os.path.join('DETR', 'data_csv')
    sub_directory = os.listdir(csv_directory)
    content_lines = []
    
    for item in sub_directory:
        subdirectory_path = os.path.join(csv_directory, item)
        name_file = os.listdir(subdirectory_path)
        st.title(f'Tablas encontradas en el Archivo : {item}')
        for x in name_file:
            
            if x.endswith('.csv'):
                file_path = os.path.join(subdirectory_path, x)
                
                #leo y muestro los csv
                with open(file_path, mode='r', newline='', encoding='utf-8') as archivo_csv:
                    lector_csv = csv.DictReader(archivo_csv)
                    for fila in lector_csv:
                        image_path = fila["image_path"]
                        content = fila["content"]
                        # caption = fila['caption']
                        # content_lines.append([caption, content])
                        st.image(image_path)
                        st.write(content)
                
            
                        
                        
                #Botón de descarga para el CSV
                with open(file_path, 'r', encoding='utf-8') as archivo_csv:
                    csv_content = archivo_csv.read()
                    st.download_button(
                        label="Descargar CSV",
                        data=csv_content,
                        file_name=os.path.basename(file_path),
                        mime='text/csv',
                    )
                    
           








    


    