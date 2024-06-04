import os
from llama_index.readers.smart_pdf_loader import SmartPDFLoader


from llmsherpa.readers.layout_reader  import Document
from llmsherpa.readers import LayoutPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.readers.file import (
    # PDFReader
    PyMuPDFReader
)

from BD.db_pinecone import conf_pinecone




path_pdf = os.path.join("data_pdf")


#construcción del indice en Pinecone con los chunks 
def create_documents(pc):
    
    Settings.chunk_size = 512
    Settings.chunk_overlap = 100
    
    pinecone_index  = pc.Index('agrichat')
    
    # namespace = '' # default namespace
    
    # pc.delete_index('agrichat')
    
    # conf_pinecone()
    
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-small",
                                        api_key=os.environ.get("OPENAI_API_KEY"),
                                        dimensions=1536)
    
            
    documents = SimpleDirectoryReader(input_dir=path_pdf).load_data()
    
    # parser = PyMuPDFReader()
    
    # file_extractor = {".pdf": parser}
    # documents = SimpleDirectoryReader(
    #         path_pdf, 
    #         file_extractor=file_extractor).load_data()
    # print(documents)

    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            
        # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
                vector_store=vector_store
        )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
            
    VectorStoreIndex.from_documents(
        documents= documents, 
        storage_context=storage_context,
        service_context=service_context
        )
    
    print('Indice con documentos creados')
    
    

# traducción de query a embedding
def query_engine_open_ai(pc, query_user):
    print("aqui")
    pinecone_index  = pc.Index('agrichat')
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=6)

    nodes = retriever.retrieve(query_user)
    print([i.get_content() for i in nodes])
    context_augmented= [i.get_content() for i in nodes]
    
    return context_augmented
    