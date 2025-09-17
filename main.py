# main.py

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# --- SETUP ---

# Load API Key
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in Streamlit Secrets or your .env file.")
    st.stop()

# Initialize session state variables if they don't exist
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []

# --- STREAMLIT APP UI ---

st.title('News Research Analyst 📈')

# --- SIDEBAR FOR URL INPUT ---
st.sidebar.title('News Article URLs')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}', key=f'url_{i}')
    if url:
        urls.append(url)

process_button_clicked = st.sidebar.button('Process URLs')

# --- MAIN CONTENT AREA ---

if process_button_clicked:
    if not urls:
        st.sidebar.warning("Please enter at least one URL.")
    else:
        with st.spinner("Processing URLs... This may take a moment."):
            try:
                # 1. Load the data from URLs
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                # 2. Split the documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=['\n\n', '\n', ',', '.'])
                texts = text_splitter.split_documents(data)

                # 3. Create embeddings and the vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(texts, embeddings)
                
                # 4. Initialize the LLM
                llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.4, groq_api_key=api_key)
                
                # 5. Create the retrieval chain and store it in session state
                st.session_state.chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
                st.session_state.processed_urls = urls
                
                st.sidebar.success("URLs processed successfully! You can now ask questions.")

            except Exception as e:
                st.sidebar.error(f"An error occurred during processing: {e}")

# Display the question input only if URLs have been processed
if st.session_state.chain is not None:
    st.info(f"Ready to answer questions about: {', '.join(st.session_state.processed_urls)}")
    
    query = st.text_input('Enter your question:')

    if query:
        with st.spinner("Searching for answers..."):
            try:
                response = st.session_state.chain({'question': query}, return_only_outputs=True)
                st.header('Answer')
                st.write(response['answer'])
                st.subheader('Sources')
                st.write(response.get('sources', 'No sources found.'))
            except Exception as e:
                st.error(f"An error occurred while answering: {e}")
else:
    st.info("Please enter URLs in the sidebar and click 'Process URLs' to begin.")