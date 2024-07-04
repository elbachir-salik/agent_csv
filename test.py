import os
import pandas as pd
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    MWDumpLoader,
)
# Function to load and chunk Excel documents
def load_and_chunk_excel(file_path):
    loader = UnstructuredExcelLoader(file_path)
    documents = loader.load_and_split()
    return documents

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# Example function to get answers from Excel data
def get_answer_from_excel(query, excel_file):
    # Load and chunk Excel documents
    documents = load_and_chunk_excel(excel_file)

    # Initialize embeddings
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    chunk_texts = [chunk.page_content for chunk in chunks]

    vectorstores = FAISS.from_texts(chunk_texts, embedding=embeddings)
    retriever = vectorstores.as_retriever()

    # embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # custom_rag_prompt = PromptTemplate.from_template(query)

    # vectorstores = FAISS.from_texts(chunks, embedding=embeddings)
    # retriever = vectorstores.as_retriever()

    # Initialize Azure Chat OpenAI
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4",
        openai_api_version="2023-05-15",
        azure_endpoint="https://aibcp4.openai.azure.com/",
        max_tokens=1400,
        api_key="a9b5778f059648b7863c397ff8f8248a",
    )

    # Define RAG chain
    
    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | PromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        | llm
        | StrOutputParser()
    )

    # Invoke RAG chain with query
    response = rag_chain.invoke(query)
    return response

# Example usage
if __name__ == "__main__":
    # Replace with your actual Excel file path
    excel_file_path = "Tableau de réclamation Gisri, Atrait , Easy Vista, anomalies virements.xlsx"

    

    # Example query
    user_question = "Quels sont les détails des réclamations enregistrées par appels ?"

    # Get the answer from Excel data
    try:
        answer = get_answer_from_excel(user_question,excel_file_path)
        print("Answer:", answer)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
