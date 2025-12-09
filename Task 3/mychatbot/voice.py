import streamlit as st
import os
import pandas as pd
from PIL import Image
import io
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from dotenv import load_dotenv

# AI & LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

# --- 👇 PASTE YOUR KEYS HERE 👇 ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-gemini-internal"

# --- SYSTEM SETUP ---
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# --- 🚆 PAGE CONFIGURATION ---
st.set_page_config(page_title="A.I.Q RailSupport", page_icon="🚆", layout="wide")

# --- 🎨 RAILWAY THEME CSS ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #101820 0%, #2c3e50 100%); background-attachment: fixed; }
    .stChatMessage { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; }
    div[data-testid="stChatMessage"]:nth-child(odd) { background-color: rgba(0, 173, 181, 0.1); border-left: 4px solid #00ADB5; }
    div[data-testid="stChatMessage"]:nth-child(even) { background-color: rgba(255, 204, 0, 0.05); border-left: 4px solid #FFCC00; }
    h1 { color: #FFCC00 !important; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 0px #000; }
    .source-box { background-color: #161b22; border: 1px dashed #30363d; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.85em; color: #8b949e; }
    audio { width: 100%; height: 40px; }
</style>
""", unsafe_allow_html=True)

# --- 🧠 INITIALIZE RESOURCES ---
@st.cache_resource
def get_rag_resources():
    # 1. Embeddings & VectorStore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # 2. LLM (Using Flash for speed & Multimodal support)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return retriever, llm

try:
    retriever, llm = get_rag_resources()
except Exception as e:
    st.error(f"⚠️ System Boot Error: {e}")
    st.stop()

# --- 📂 FILE PROCESSING FUNCTIONS ---
def process_uploaded_file(uploaded_file):
    """Extracts text from PDF, TXT, CSV"""
    try:
        if uploaded_file.type == "application/pdf":
            # Save temp for PyMuPDF
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyMuPDFLoader("temp_upload.pdf")
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
            
    except Exception as e:
        return f"Error reading file: {e}"
    return ""

def process_audio(audio_bytes):
    """Transcribe Voice"""
    r = sr.Recognizer()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes.read())
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = r.record(source)
        try:
            return r.recognize_google(audio_data)
        except:
            return None

def text_to_speech(text):
    """Generate Announcer Audio"""
    try:
        tts = gTTS(text="Attention please. " + text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except:
        return None

# --- 🎛️ SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Control Center")
    st.info("System Status: **ONLINE** 🟢")
    enable_voice = st.toggle("📢 Enable Voice Output", value=True)
    
    st.markdown("### 📎 Attachments")
    uploaded_file = st.file_uploader("Upload Ticket/Image/Doc", type=["png", "jpg", "jpeg", "pdf", "csv", "txt"])
    
    if st.button("🧹 Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- 🚅 MAIN INTERFACE ---
st.title("🚆 A.I.Q RailSupport")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    role = message["role"]
    avatar = "🧑‍✈️" if role == "user" else "🚆"
    with st.chat_message(role, avatar=avatar):
        # Check if content is a list (multimodal) or string
        if isinstance(message["content"], list):
            # It's an image message, just show the text part
            text_part = next((item for item in message["content"] if item["type"] == "text"), None)
            if text_part: st.markdown(text_part["text"])
            # We could also show the image if stored, but keep it simple
        else:
            st.markdown(message["content"])

# --- 🎤 INPUT & LOGIC ---
voice_val = st.audio_input("🎤 Press to Speak")
text_val = st.chat_input("Type your query...")

user_query = None
if voice_val:
    with st.spinner("🎧 Listening..."):
        user_query = process_audio(voice_val)
elif text_val:
    user_query = text_val

if user_query:
    # 1. Show User Input
    st.chat_message("user", avatar="🧑‍✈️").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Prepare Context (RAG + Uploads)
    with st.chat_message("assistant", avatar="🚆"):
        with st.spinner("🔄 Processing signals..."):
            try:
                # A. Retrieve RAG Context
                rag_docs = retriever.invoke(user_query)
                rag_context = "\n".join([doc.page_content for doc in rag_docs])
                sources = {os.path.basename(d.metadata.get("source", "Unknown")) for d in rag_docs}

                # B. Handle File Uploads
                file_context = ""
                image_message_part = None

                if uploaded_file:
                    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                        # Process Image for Gemini
                        image = Image.open(uploaded_file)
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format=image.format)
                        image_data = image_bytes.getvalue()
                        
                        # Create Image Block
                        import base64
                        b64_img = base64.b64encode(image_data).decode()
                        image_message_part = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        }
                        st.image(uploaded_file, caption="Attached Image", width=300)
                        file_context = "User has attached an image. Analyze it alongside the query."
                    else:
                        # Process Text Docs
                        file_context = f"\n\n--- ATTACHED FILE CONTENT ---\n{process_uploaded_file(uploaded_file)}\n---------------------------------\n"
                        st.success(f"📎 Reading attached file: {uploaded_file.name}")

                # C. Construct Final Prompt
                system_instruction = (
                    "You are 'A.I.Q RailSupport', an expert railway assistant. "
                    "Answer based on the Internal Database Context AND any Attached Files. "
                    "If an image is attached, describe it or answer the user's specific question about it. "
                    "Keep answers professional and precise."
                    f"\n\nInternal Database Context:\n{rag_context}"
                    f"{file_context}"
                )

                # D. Build Message Payload
                messages = [SystemMessage(content=system_instruction)]
                
                user_content = [{"type": "text", "text": user_query}]
                if image_message_part:
                    user_content.append(image_message_part)
                
                messages.append(HumanMessage(content=user_content))

                # E. Invoke LLM
                ai_response = llm.invoke(messages)
                answer = ai_response.content

                # F. Display & Voice
                st.markdown(answer)
                if sources:
                    st.markdown(f"<div class='source-box'><strong>🎫 Reference Sources:</strong><br>{', '.join(sources)}</div>", unsafe_allow_html=True)
                
                if enable_voice:
                    audio_fp = text_to_speech(answer)
                    if audio_fp:
                        st.audio(audio_fp, format='audio/mp3', start_time=0, autoplay=True)

                # Save History
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"⚠️ Signal Error: {e}")