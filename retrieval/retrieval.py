from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding


import os


def retrieval(pinecone_index):
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-small",
                                  api_key=os.environ.get("OPENAI_API_KEY"),
                                  dimensions=1536)
    
    namespace = '' # default namespace

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    
#     index = VectorStoreIndex.from_documents(
#         #documents,
#         storage_context=storage_context,
#         service_context=service_context
# )
    
