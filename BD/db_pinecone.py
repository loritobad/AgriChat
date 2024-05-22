from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



def conf_pinecone():
   
    
    # Crear una instancia de Pinecone con tu clave API
    pc = Pinecone(
    api_key_pinecone=os.environ.get("PINECONE_API_KEY"),  # Puedes usar dotenv o métodos similares para obtener la clave
    environment=os.environ.get("PINECONE_ENVIRONMENT")  # El entorno de Pinecone
    )


    #indice en BD
    index_name = "agrichat"
    
    if index_name not in pc.list_indexes().names(): # si no existe el indice lo crea
        pc.create_index(
            name=index_name,
            dimension=1536,  # Ajusta según las dimensiones de tus vectores
            metric='cosine',  # Ajusta según el tipo de métrica que necesitas
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
    ) 
        )
    
    
    
    return pc


    
    

    