
import os, json
import pandas as pd
from ragas import evaluate
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall, answer_correctness

from datasets import Dataset

import streamlit as st

from langchain.schema import HumanMessage

from BD.db_pinecone import conf_pinecone


from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

distributions = {
        simple: 0.2,
        reasoning: 0.2,
        multi_context: 0.4,
        conditional: 0.2
    }



def read_pdfs_from_directory():
    directory_path = os.path.join('evaluation', 'data', '11530.pdf')
    loader = PyMuPDFLoader(directory_path)
    documents = loader.load()
    for document in documents:
        document.metadata['filename'] = document.metadata['source']
    return documents

def query_pinecone(embedding):
    pc = conf_pinecone()
    
    index = pc.Index('agrichat')
    
    #o
    query_response = index.query(vector=[embedding],
                                 top_k=4,
                                 include_metadata= True)
    contexts = []
    
    for match in query_response['matches']:
        metadata_str = match['metadata']['_node_content']
        metadata = json.loads(metadata_str)
        text = metadata.get('text', '')
        contexts.append(text)
    
    
    return contexts
    
   

def generate_custom_testset(documents, num_tests=1, distributions=None):
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-0125",
                               api_key=api_key,
                               temperature=0.5,
                               max_tokens=200
                               )
    critic_llm = ChatOpenAI(model="gpt-4",
                            api_key=api_key,
                            temperature=0.5,
                            max_tokens=200)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                                  api_key=api_key,
                                  dimensions=1536)

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )
    
    language = "spanish"
    evolutions = list(distributions.keys())
    generator.adapt(language, evolutions=evolutions)
    generator.save(evolutions=evolutions)

    # # Generar pruebas personalizadas
    testset = generator.generate_with_langchain_docs(documents, num_tests, distributions)
    
    results = []
   
    
    for entry in testset._to_records():
    
        question = entry['question']
        contexts = entry['contexts']
        ground_truth = entry['ground_truth']
        type_cuestion = entry['evolution_type']
        
        
        #generar embedding de preguntas
        embedding = embeddings.embed_query(question)
    
        contexts = query_pinecone(embedding)

        # Generar la respuesta utilizando el modelo generador
        context_combined = " ".join(contexts) + "\n\nPregunta: " + question
        response = generator_llm.generate([[HumanMessage(content=context_combined)]])
        answer = response.generations[0][0].text
        
        results.append({
            'question': question,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'answer': answer,
            'type_cuestion': type_cuestion
        })
    
    testset = Dataset.from_list(results)
    
    return testset


def evaluate_generation():
    documents = read_pdfs_from_directory()
    # generate_custom_testset(documents, num_tests=1, distributions=distributions)
    testset_df = generate_custom_testset(documents, num_tests=2, distributions=distributions)
    
    eval_results = evaluate(
        testset_df,
        metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness
      ],
        llm=ChatOpenAI(model="gpt-4", api_key=api_key)
    )
    
    eval_results = eval_results.to_pandas()
    
    eval_results.to_csv(os.path.join('evaluation', 'data', "evaluation_results.csv"),
                        index=False,
                        encoding='utf-8',
                        quotechar='"')
    
    st.session_state.eval_results = eval_results
    
    return eval_results

  