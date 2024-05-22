import os
import pandas as pd

from ragas import evaluate
from langchain_community.document_loaders.directory import DirectoryLoader

from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser

from langchain_community.document_loaders.pdf import PyMuPDFLoader

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context,conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)

def read_pdfs_from_directory():
    
    directory_path = os.path.join('evaluation', 'data','11854.pdf')
    
    
    loader = PyMuPDFLoader(directory_path)
   
    documents = loader.load()
    
    for document in documents:
        document.metadata['filename'] = document.metadata['source']
    

    return documents

def generated_sample_sintetic():
    
    documents = read_pdfs_from_directory()
    api_key = os.getenv("OPENAI_API_KEY")
    

    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)
    critic_llm = ChatOpenAI(model="gpt-4", api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # Adaptar al idioma
    language = "spanish"
    generator.adapt(language, evolutions=[simple, reasoning, conditional, multi_context])
    generator.save(evolutions=[simple, reasoning, multi_context, conditional])
    
    distributions = {
        simple: 0.4,
        reasoning: 0.2,
        multi_context: 0.2,
        conditional: 0.2
    }
    
    # Generar el conjunto de datos de prueba
    testset = generator.generate_with_langchain_docs(documents, 1, distributions, with_debugging_logs=True)
    
    return testset

# def convert_testset_to_dataframe(testset):
#     data = []
#     for data_row in testset.test_data:
#         question = data_row.question
#         contexts = " ".join(data_row.contexts)
#         ground_truth = data_row.ground_truth
#         data.append({"question": question, "contexts": contexts, "ground_truth": ground_truth})
    
#     testset_df = pd.DataFrame(data)
#     return testset_df

def evaluate_generation():
    testset = generated_sample_sintetic()

    
    eval_results = evaluate(
        testset,
        metrics=[
        context_precision,
        faithfulness,
        answer_relevancy
      ],
    )
    
    return eval_results
