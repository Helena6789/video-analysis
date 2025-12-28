# tools/init_policy_vector_db.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Define where to save the database
VECTOR_DB_DIR = "knowledge_base/policy_vector_store"

def init_vector_db():
    # 1. Load the Policy Text
    loader = TextLoader("knowledge_base/policy_123.txt")
    documents = loader.load()

    # 2. Split text into chunks (e.g., by paragraph or section)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings (Turn text into numbers)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create and Save the Vector Store locally
    if os.path.exists(VECTOR_DB_DIR):
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)
        
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_DIR
    )
    
    print(f"✅ Vector DB created with {len(docs)} chunks at {VECTOR_DB_DIR}")

if __name__ == "__main__":
    init_vector_db()