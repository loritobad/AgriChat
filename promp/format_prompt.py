from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI

import os
from llama_index.core import Settings

Settings.llm_openai = OpenAI(model="gpt-3.5-turbo-0125",
                   temperature=0.3,
                   api_key= os.environ.get("OPENAI_API_KEY"))

# Settings.llm_mixtral_hf = HuggingFaceLLM(model_name= 'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1')




# vemos el formato del prompt que enviamos al llm
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


def prompt_openai(pc, contex_augmented, query_user):
    
    # login(token = os.environ.get("HF_API_KEY "))
    pinecone_index  = pc.Index('agrichat')
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
     
    #promp template
    
    template = (
        "Answer the question in Spanish."
        "Context information is below."
        "---------------------"
        "{context_str}"
        "---------------------"
        "Given the context information and not prior knowledge, answer the query."
        "Query: {query_str}"  
    )
    
    qa_template = PromptTemplate(template)
    
    prompt = qa_template.format(context_str = contex_augmented, query_str = query_user)
    
    query_engine = index.as_query_engine(similarity_top_k=2, llm=Settings.llm_openai)
    
    #AÑADIR EL PROMPT DE LOS .TXT
    
    #DEVUELVO EL PROMPT RECUPERADO Y AÑADO EL TXT
    response = query_engine.query(prompt)
    print('RESPUESTA', str(response))
    return response
    
   
    
    

    
    
        
