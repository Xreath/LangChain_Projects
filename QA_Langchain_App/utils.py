import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tiktoken
import streamlit as st


def load_document(file):
    _, ext = os.path.splitext(file)

    supported_formats = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt" : TextLoader}

    if ext in supported_formats:
        loader_class = supported_formats[ext]
        print(f"[INFO] Loading {file}")
        loader = loader_class(file)
    else:
        print("[INFO] Document format is not supported!")
        return None

    data = loader.load()
    return data

def chunk_data(data,chunk_size=256,chunk_overlap=20):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks=text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings=OpenAIEmbeddings()
    vector_store=Chroma.from_documents(chunks,embeddings)
    return vector_store


def ask_and_get_answer(vector_store,q):
    llm= ChatOpenAI(model="gpt-3.5-turbo",temperature=0.8)
    
    retriever=vector_store.as_retriever(search_type="similarity")
    chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)
    
    return chain.run(q)
    

def print_embedding_cost(texts):
    enc=tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens=sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"[INFO] Total Tokens: {total_tokens}")
    print(f"[INFO] Embedding Cost in $: {total_tokens / 1000*0.0004:.6f}") 
    


    