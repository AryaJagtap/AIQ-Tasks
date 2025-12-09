import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

# --- 👇 PASTE YOUR KEYS HERE 👇 ---
# We ONLY need Pinecone here. Google/HF keys are not used for Local Ingestion.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-gemini-internal"

# --- SYSTEM SETUP ---
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def load_documents_from_folder(folder_path):
    """Loads PDF, CSV, TXT, and JSON files from a folder."""
    documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"⚠️ Created missing folder: {folder_path}")
        return []

    print(f"📂 Scanning '{folder_path}' for files...")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                print(f"   📄 Loading PDF: {filename}")
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".csv"):
                print(f"   📊 Loading CSV: {filename}")
                loader = CSVLoader(file_path, encoding="utf-8") 
                documents.extend(loader.load())
            elif filename.endswith(".txt"):
                print(f"   📝 Loading TXT: {filename}")
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            elif filename.endswith(".json"):
                print(f"   📦 Loading JSON: {filename}")
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception as e:
            print(f"   ⚠️ Error loading {filename}: {e}")

    return documents

def load_and_process_data():
    # 1. Load Data
    docs = load_documents_from_folder('./data')
    if not docs:
        print("❌ No compatible files found in 'data' folder.")
        return

    print(f"✅ Loaded {len(docs)} document chunks.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"✂️ Final split count: {len(splits)} chunks.")

    # 3. Initialize Embeddings (LOCAL - NO LIMITS)
    print("🧠 Loading Local Embeddings (HuggingFace)...")
    # This downloads the model to your PC once, then runs offline.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Setup Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"📦 Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # The dimension for 'all-MiniLM-L6-v2' is 384
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(2)
    else:
        # Check if existing index has correct dimension
        idx = pc.Index(INDEX_NAME)
        stats = idx.describe_index_stats()
        if stats.get('dimension') != 384:
            print("⚠️ WARNING: Existing index has wrong dimensions for this model.")
            print("   Please delete the index 'rag-gemini-internal' in Pinecone Console and restart.")
            return
        print(f"ℹ️ Index {INDEX_NAME} exists and is ready.")

    # 5. Upload to Pinecone
    # Since we are running locally, we don't have a rate limit! We can go faster.
    batch_size = 200
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    print(f"🚀 Uploading {len(splits)} chunks to Pinecone...")
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        print(f"   🔹 Batch {i//batch_size + 1}...", end=" ")
        try:
            vectorstore.add_documents(batch)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("✅ Data ingestion complete! You can now run app.py")

if __name__ == "__main__":
    load_and_process_data()