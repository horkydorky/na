# build_vectorstore.py

import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Updated imports to use langchain_community
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- IMPORTANT ---
# --- PASTE THE URLs YOU WANT TO PROCESS HERE ---
urls = [
    "https://www.cnbc.com/2024/07/22/how-the-paris-olympics-opening-ceremony-will-be-different-this-year.html",
    "https://www.reuters.com/sports/olympics/d-day-paris-olympics-organisers-keep-fingers-crossed-opening-ceremony-2024-07-26/",
    "https://apnews.com/article/paris-olympics-2024-opening-ceremony-f30560938a16511b81561543b593457a"
]

print("Loading data from URLs...")
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=['\n\n', '\n', ',', '.'])
texts = text_splitter.split_documents(data)

print("Creating embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

print("Saving vector store to vectorstore.pkl...")
with open('vectorstore.pkl', 'wb') as f:
    pickle.dump(vectorstore, f)

print("--- FINISHED! vectorstore.pkl is created. ---")