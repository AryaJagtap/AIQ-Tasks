import streamlit as st
import os
import pandas as pd
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# --- 1. CHANGED: IMPORT GROQ INSTEAD OF GOOGLE ---
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

# ---  PASTE YOUR KEYS HERE  ---
# I have renamed this variable to GROQ_API_KEY for clarity
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-gemini-internal"

# --- SYSTEM SETUP ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# --- 🚆 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="A.I.Q RailSupport",
    page_icon="🚆",
    layout="wide"
)

# --- 🎨 RAILWAY THEME CSS (KEPT EXACTLY AS IS) ---
st.markdown("""
<style>
    /* 1. Main Background - Steel Railway Gradient */
    .stApp {
        background: linear-gradient(135deg, #101820 0%, #2c3e50 100%);
        background-attachment: fixed;
    }

    /* 2. Chat Message Bubbles */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05); /* Glass effect */
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* User Message (Passenger) - Subtle Blue Tint */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(0, 173, 181, 0.1);
        border-left: 4px solid #00ADB5;
    }

    /* Bot Message (Train) - Yellow/Green */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(255, 204, 0, 0.05);
        border-left: 4px solid #FFCC00;
    }

    /* 3. Sidebar Styling - Control Panel Look */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 2px solid #1f6feb;
    }
    
    /* 4. Headers */
    h1 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #FFCC00 !important; /* Signal Yellow */
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 0px #000;
    }
    
    h3 {
        color: #aeb9cc !important;
    }

    /* 5. Sources Box styling */
    .source-box {
        background-color: #161b22;
        border: 1px dashed #30363d;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.85em;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# --- 📂 FILE PROCESSING LOGIC ---
def process_uploaded_file(uploaded_file):
    """Helper to extract text from different file types"""
    try:
        # PDF Handling
        if uploaded_file.type == "application/pdf":
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyMuPDFLoader("temp_upload.pdf")
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        
        # Text File Handling
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        
        # CSV Handling
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
            
    except Exception as e:
        return f"Error reading file: {e}"
    return ""

# --- 🧠 INITIALIZE RAG SYSTEM ---
@st.cache_resource
def initialize_rag_system():
    # 1. Embeddings (Local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Connect to Pinecone
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    # 3. Setup LLM - CHANGED TO GROQ
    # We use 'llama-3.2-11b-vision-preview' because it supports IMAGES. 
    # The standard 'llama-3.1-8b' is text-only and would break image uploads.
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant", 
        temperature=0.3
    )
    
    # 4. Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # 5. Prompt
    system_prompt = (
        "You are 'A.I.Q RailSupport', an expert railway assistant. "
        "Use the retrieved context to answer the passenger's question. "
        "If the answer is not in the documents, politely state that you don't have that information. "
        "Keep answers professional, precise, and helpful."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain, retriever, llm

# Start System
try:
    rag_chain, retriever, llm = initialize_rag_system()
except Exception as e:
    st.error(f"⚠️ Connection Signal Lost: {e}")
    st.stop()

# --- 🎛️ SIDEBAR CONTROL PANEL ---
with st.sidebar:
    # Use markdown for smaller header to fit in one line
    st.markdown("## 🎛️ Control Center")
    st.markdown("---")
    st.info("System Status: **ONLINE** 🟢")
    
    # --- FILE UPLOADER ---
    st.markdown("### 📎 Attachments")
    uploaded_file = st.file_uploader("Upload Image, PDF, TXT, CSV", type=["png", "jpg", "jpeg", "pdf", "csv", "txt"])
    # ---------------------

    st.markdown("### 🛠️ Operations")
    if st.button("🧹 Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("A.I.Q RailSupport v1.1 (Groq)")

# --- 🚅 MAIN INTERFACE ---
st.title("🚆 A.I.Q RailSupport")
st.markdown("### 🎫 Intelligent Passenger Assistance System")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    role = message["role"]
    avatar = "🧑‍✈️" if role == "user" else "🚆"
    with st.chat_message(role, avatar=avatar):
        # Handle if content is a list (multimodal history)
        if isinstance(message["content"], list):
             text_content = next((item["text"] for item in message["content"] if item["type"] == "text"), "")
             st.markdown(text_content)
        else:
            st.markdown(message["content"])

# Handle Input
if user_input := st.chat_input("Enter your query here (e.g., 'What are the safety protocols?')..."):
    # 1. Show User Message
    st.chat_message("user", avatar="🧑‍✈️").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Generate Answer
    with st.chat_message("assistant", avatar="🚆"):
        with st.spinner("🔄 Retrieving signals from database..."):
            try:
                # --- LOGIC FOR FILE HANDLING ---
                
                # Case A: User uploaded an IMAGE
                if uploaded_file and uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                    # 1. Retrieve text context manually
                    rag_docs = retriever.invoke(user_input)
                    rag_context = "\n".join([doc.page_content for doc in rag_docs])
                    
                    # 2. Process Image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Attached Image", width=300)
                    
                    # Convert to Base64 
                    buffered = io.BytesIO()
                    image.save(buffered, format=image.format)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # 3. Create Multimodal Message (Supported by Llama 3.2 Vision)
                    system_text = f"You are A.I.Q RailSupport. Analyze the image and the context provided.\nContext from DB: {rag_context}"
                    
                    messages = [
                        SystemMessage(content=system_text),
                        HumanMessage(content=[
                            {"type": "text", "text": user_input},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ])
                    ]
                    
                    # 4. Invoke LLM directly
                    ai_response = llm.invoke(messages)
                    answer = ai_response.content
                    
                    # Get sources for display
                    unique_sources = set()
                    for doc in rag_docs:
                        unique_sources.add(os.path.basename(doc.metadata.get("source", "Unknown")))

                # Case B: User uploaded a DOCUMENT (PDF/TXT/CSV) or NO FILE
                else:
                    file_context = ""
                    if uploaded_file:
                        st.info(f"📎 Analyzing attached file: {uploaded_file.name}")
                        extracted_text = process_uploaded_file(uploaded_file)
                        file_context = f"\n\n[ATTACHED FILE CONTENT]:\n{extracted_text}\n"

                    # Combine user input with file context for the RAG chain
                    final_input = user_input + file_context
                    
                    response = rag_chain.invoke({"input": final_input})
                    answer = response["answer"]
                    
                    # Extract Sources from RAG
                    sources = response["context"]
                    unique_sources = set()
                    for doc in sources:
                        source_path = doc.metadata.get("source", "Unknown")
                        filename = os.path.basename(source_path)
                        page = doc.metadata.get("page", None)
                        if page:
                            unique_sources.add(f"{filename} (Pg {int(page)+1})")
                        else:
                            unique_sources.add(filename)

                # -----------------------------------

                # Display Answer
                st.markdown(answer)
                
                # Display Sources in a special "Ticket" styled box
                if unique_sources:
                    source_list = ', '.join(unique_sources)
                    st.markdown(f"""
                        <div class="source-box">
                            <strong>🎫 Reference Sources:</strong><br>
                            {source_list}
                        </div>
                    """, unsafe_allow_html=True)

                # Save to history
                final_content = answer + (f"\n\n**🎫 Sources:** {', '.join(unique_sources)}" if unique_sources else "")
                st.session_state.messages.append({"role": "assistant", "content": final_content})
                
            except Exception as e:
                st.error(f"⚠️ Signal Error: {e}")