# build_vectorstore.py
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Put the URLs you want to process here
urls = [
    "https://example-news-url.com/article1",
    "https://another-example.com/article2",
    "https://some-other-news.com/article3"
]

# 1. Load the data
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# 2. Split the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=['\n\n', '\n', ',', '.'])
texts = text_splitter.split_documents(data)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build and save the FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
with open('vectorstore.pkl', 'wb') as f:
    pickle.dump(vectorstore, f)

print("Vector store created successfully and saved to vectorstore.pkl")