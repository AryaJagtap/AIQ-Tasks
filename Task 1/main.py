import os
import time

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "pdf-rag-store"
PDF_PATH = "CSL.pdf"


def initialize_pinecone():
    """Initialize Pinecone and create/get index"""
    print("=" * 70)
    print("RAG System with Pinecone, LangChain & Google Gemini")
    print("=" * 70)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(
            f"Index '{INDEX_NAME}' already exists. Deleting to recreate with correct dimensions..."
        )
        pc.delete_index(INDEX_NAME)
        time.sleep(5)  # Wait for deletion to complete

    # Create new index with 3072 dimensions for gemini-embedding-001
    print(f"🔧 Creating new Pinecone index: {INDEX_NAME} with dimension 3072")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # Dimension for gemini-embedding-001
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("⏳ Waiting for index to be ready...")
        time.sleep(1)

    print("✓ Index created and ready!")
    return pc


def load_and_process_pdf(pdf_path):
    """Load and split PDF into chunks"""
    print(f"\n Processing PDF: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"✓ Loaded {len(documents)} pages from PDF")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    print(f"✓ Created {len(chunks)} text chunks")
    return chunks


def create_vectorstore(chunks):
    """Create embeddings and store in Pinecone"""
    print("\n📤 Creating embeddings with gemini-embedding-001...")

    # Initialize embeddings with gemini-embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY
    )

    # Create vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=INDEX_NAME
    )

    print("✓ Embeddings stored in Pinecone successfully!")
    return vectorstore, embeddings


def create_qa_chain(vectorstore):
    """Create QA chain with retrieval"""
    print("\n Setting up QA chain with Google Gemini...")

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0.3
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    print("✓ QA chain ready!")
    return qa_chain


def ask_question(qa_chain, question):
    """Ask a question and get answer"""
    print("\n" + "=" * 70)
    print(f" Question: {question}")
    print("=" * 70)

    result = qa_chain.invoke({"query": question})

    print(f"\n Answer:\n{result['result']}")

    print(f"\n Source Documents ({len(result['source_documents'])} chunks used):")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")


def main():
    try:
        # Step 1: Initialize Pinecone
        pc = initialize_pinecone()

        # Step 2: Load and process PDF
        chunks = load_and_process_pdf(PDF_PATH)

        # Step 3: Create embeddings and store in Pinecone
        vectorstore, embeddings = create_vectorstore(chunks)

        # Step 4: Create QA chain
        qa_chain = create_qa_chain(vectorstore)

        # Step 5: Interactive Q&A loop
        print("\n" + "=" * 70)
        print("System ready! Ask questions about your PDF")
        print("=" * 70)
        print("Type 'exit' or 'quit' to stop\n")

        while True:
            question = input("\n🔍 Your question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("\n Goodbye!")
                break

            if not question:
                print("Please enter a question")
                continue

            ask_question(qa_chain, question)

    except Exception as e:
        print(f"\n Error: {e}")
        print("\nPlease ensure:")
        print("  1. PINECONE_API_KEY is set in your .env file")
        print("  2. GOOGLE_API_KEY is set in your .env file")
        print("  3. PDF file path is correct")
        print("  4. All required packages are installed:")
        print("     pip install pinecone-client langchain langchain-google-genai")
        print("     pip install langchain-pinecone pypdf python-dotenv")


if __name__ == "__main__":
    main()
