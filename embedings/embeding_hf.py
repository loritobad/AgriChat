from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import os


embed_model = OpenAIEmbedding(model="text-embedding-3-small",#text-embedding-3-small
                                  api_key=os.environ.get("OPENAI_API_KEY"),
                                  embed_batch_size=10,
                                  dimensions=1536)
Settings.embed_model = embed_model
       

def generate_embedings_opeanAI_query(query):
    
    query_embedding = embed_model.get_query_embedding(query)
    print('Embedding de consulta hecho')
    
    return query_embedding


