from streamlit_elements import elements, mui, html
import streamlit as st
import os
import pandas as pd
import csv

from evaluation.eval_ragas import evaluate_generation


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
    evaluation =  st.container(height=700, border=True)
     
with evaluation:
    
    st.title("Evaluación LLM")
    
    if st.button("Evaluar"):
        eval_generation = evaluate_generation()
        st.write(eval_generation)
   