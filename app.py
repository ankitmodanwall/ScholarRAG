import os
import streamlit as st
from dotenv import load_dotenv

# --- CORE ENGINE ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIG & INTERACTIVE RGB UI ---
load_dotenv()
GROQ_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
BASE_WS_DIR = "workspaces"

st.set_page_config(page_title="ScholarRAG Elite", page_icon="🤖", layout="wide")

# CSS for Social Icons, Mobile Gap Fix, and RGB Glow
st.markdown("""
    <style>
    /* 1. Remove Triple Dots & Default UI */
    #MainMenu, header, footer, .stDeployButton {visibility: hidden !important; display: none !important;}
    
    /* 2. RGB Glow & Global Styling */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Social Bar (Top Right) */
    .social-bar {
        position: fixed;
        top: 15px;
        right: 25px;
        display: flex;
        gap: 15px;
        z-index: 10001;
    }
    .social-icon {
        color: #f1c40f; /* Yellow Base */
        font-size: 22px;
        transition: 0.3s;
        text-decoration: none !important;
    }
    .social-icon:hover {
        color: #e74c3c; /* Red Hover Mix */
        transform: scale(1.2);
    }

    /* Interactive Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #f1c40f, #e74c3c);
        color: white; border: none; transition: 0.3s;
        border-radius: 10px;
    }
    div.stButton > button:first-child:hover { transform: scale(1.02); box-shadow: 0 0 15px rgba(231, 76, 60, 0.6); }

    /* Header Line with RGB Gradient */
    .header-line {
        border-bottom: 2px solid;
        border-image: linear-gradient(to right, #f1c40f, #e74c3c) 1;
        padding-bottom: 12px;
        margin-bottom: 25px;
    }

    /* 3. MOBILE OPTIMIZATION */
    @media (max-width: 768px) {
        .social-bar { top: 10px; right: 15px; gap: 10px; }
        .social-icon { font-size: 18px; }
        .main .block-container { padding-top: 15px !important; }
        .stVerticalBlock { gap: 0.4rem !important; }
    }
    
    .welcome-section {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 55vh; text-align: center;
    }
    
    .main .block-container { max-width: 850px !important; margin: auto; }
    </style>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <div class="social-bar">
        <a href="https://linkedin.com/in/ankit-modanwal" target="_blank" class="social-icon"><i class="fab fa-linkedin"></i></a>
        <a href="https://github.com/ankitmodanwall" target="_blank" class="social-icon"><i class="fab fa-github"></i></a>
        <a href="https://ankie200.substack.com" target="_blank" class="social-icon"><i class="fab fa-telegram"></i></a>
    </div>
""", unsafe_allow_html=True)

# --- 2. THE PINNED RGB HEADER ---
if "messages" not in st.session_state: st.session_state.messages = []

with st.container():
    st.markdown('<div class="header-line">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 3.5, 1, 1])
    with c1: lang = st.selectbox("🌐", ["Hinglish", "English", "Hindi"], label_visibility="collapsed")
    with c2: voice_on = st.toggle("📢", value=False)
    with c3: uploads = st.file_uploader("Upload Notes", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    with c4:
        if st.button("🚀 SYNC", use_container_width=True):
            if uploads:
                with st.status("Absorbing Material..."):
                    ws_path = os.path.join(BASE_WS_DIR, "General")
                    if not os.path.exists(ws_path): os.makedirs(ws_path, exist_ok=True)
                    all_docs = []
                    for f in uploads:
                        temp = f"temp_{f.name}"
                        with open(temp, "wb") as t: t.write(f.getbuffer())
                        try:
                            loader = PyPDFLoader(temp)
                            all_docs.extend(loader.load())
                        finally: os.remove(temp)
                    if all_docs:
                        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(all_docs)
                        if chunks:
                            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            FAISS.from_documents(chunks, embed).save_local(ws_path)
                            st.rerun()
    with c5:
        if st.button("🗑️ CLEAR", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 3. WELCOME & CHAT ---
if not st.session_state.messages:
    # Market-Ready Welcome
    st.markdown("""
        <div class="welcome-section">
            <h1 style="background: linear-gradient(to right, #f1c40f, #e74c3c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight:700;">How can I help you learn?</h1>
            <p style="opacity: 0.6; font-size: 1.1rem; max-width: 550px; margin-top: 10px;">
                Your professional study OS is ready. Upload materials in the header to begin.
            </p>
        </div>
    """, unsafe_allow_html=True)

for m in st.session_state.messages:
    bubble_style = "background:#161b22; border:1px solid #30363d; border-radius:12px; padding:15px; margin-bottom:12px;"
    st.markdown(f'<div style="{bubble_style}">{m["content"]}</div>', unsafe_allow_html=True)

# --- 4. EXECUTION ---
if prompt := st.chat_input("Ask a question about your study material..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        loading = st.empty()
        loading.markdown("⏳ *Scholar is thinking...*")
        
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_KEY)
        
        ws_path = os.path.join(BASE_WS_DIR, "General")
        ctx = ""
        if os.path.exists(os.path.join(ws_path, "index.faiss")):
            db = FAISS.load_local(ws_path, embed, allow_dangerous_deserialization=True)
            ctx = "\n".join([d.page_content for d in db.as_retriever().invoke(prompt)])
        
        ans = llm.invoke(f"Professional teacher. Explain in {lang} with Mermaid: {prompt}. Context: {ctx}").content
        loading.empty()
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()