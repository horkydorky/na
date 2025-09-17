#imports
import os
import streamlit as st
import pickle
import time

from langchain_groq import ChatGroq


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.vectorstores import FAISS

#API key setup
from dotenv import load_dotenv

load_dotenv()

#Streamlit UI

st.title('News Research Analyst')
st.sidebar.title('News Article Urls')
urls=[]

for i in range (3):
    url=st.sidebar.text_input(f'Enter News Article URL {i+1}', key=f'url_{i}')
    urls.append(url)

processed_url_clicked=st.sidebar.button('Process URLs')

main_placeholder=st.empty()
llm=ChatGroq(model='llama-3.1-8b-instant', temperature=0.4)
if processed_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, separators=['\n\n', '\n', ',', '.'])

    texts=text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(texts, embeddings)

    with open('vectorstore.pkl', 'wb') as f:
        pickle.dump(vectorstore, f)

query=main_placeholder.text_input('Enter your question')
if query:
    if os.path.exists('vectorstore.pkl'):
        with open('vectorstore.pkl', 'rb') as f:
            vectorstore=pickle.load(f)
        chain=RetrievalQAWithSourcesChain.from_chain_type(llm=llm,  retriever=vectorstore.as_retriever())
        response=chain({'question': query}, return_only_outputs=True)
        st.header('Answer')
        st.write(response['answer'])