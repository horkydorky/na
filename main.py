# main.py

import streamlit as st
import pickle
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

# This function is cached to load the model and vector store only once
@st.cache_resource
def load_assets():
    # Load the vector store
    with open('vectorstore.pkl', 'rb') as f:
        vectorstore = pickle.load(f)
    
    # Get the API key securely
    # It will try to get it from Streamlit Secrets first, then from .env for local dev
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in Streamlit Secrets or your .env file.")

    llm = ChatGroq(
        model='llama-3.1-8b-instant', 
        temperature=0.4, 
        groq_api_key=api_key
    )
    
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return chain

# --- Streamlit App UI ---
st.title('News Research Analyst 📈')
st.info("Ask a question about the pre-loaded news articles.")

try:
    chain = load_assets()
    query = st.text_input('Enter your question:')

    if query:
        with st.spinner("Searching for answers..."):
            response = chain({'question': query}, return_only_outputs=True)
            st.header('Answer')
            st.write(response['answer'])
            st.subheader('Sources')
            st.write(response.get('sources', 'No sources found.'))

except Exception as e:
    st.error(f"An error occurred: {e}")