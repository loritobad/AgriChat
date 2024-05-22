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
                                  embed_batch_size=10,
                                  dimensions=384)
    
    namespace = '' # default namespace

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    
    index = VectorStoreIndex.from_documents(
        #documents,
        storage_context=storage_context,
        service_context=service_context
)
    
    
    
    
    
    
#     pinecone_index = vector_store.Index("agrichat")

#     vector_store = PineconeVectorStore(pinecone_index=pinecone_index,
#                                        index_name='agrichat',
#                                        api_key=os.environ.get("PINECONE_API_KEY")
#                                        )

#     vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# # Grab 5 search results
#     retriever = VectorIndexRetriever(index=vector_index,
#                                      similarity_top_k=5,
#                                      embed_model= Settings.embed_model,
#                                      verbose=True)
    

#     query_engine = RetrieverQueryEngine(retriever=retriever)


#     llm_query = query_engine.query('¿Como se llama el producto')

#     print(llm_query)