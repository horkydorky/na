#imports
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found. Please add GROQ_API_KEY to your .env file.")
    st.stop()

# --- Streamlit UI ---
st.title('📰 News Research Analyst')
st.sidebar.title('News Article URLs')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f'Enter News Article URL {i+1}', key=f'url_{i}')
    urls.append(url)

urls = [u.strip() for u in urls if u.strip()]  # filter out empty ones

processed_url_clicked = st.sidebar.button('Process URLs')
main_placeholder = st.empty()

# --- LLM Setup ---
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.4)

# --- Process URLs ---
if processed_url_clicked:
    if not urls:
        st.error("⚠️ Please enter at least one valid URL.")
        st.stop()

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    st.session_state.vectorstore = vectorstore
    st.success("✅ URLs processed and knowledge base created!")

# --- Ask Questions ---
query = main_placeholder.text_input('Ask a question about the news articles:')

if query:
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please process URLs first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

        response = chain({"question": query}, return_only_outputs=True)

        st.header('Answer')
        st.write(response["answer"])

        st.subheader("Sources")
        st.write(response.get("sources", "No sources found"))