# 🤖 ScholarRAG Elite - Modern AI Study OS

ScholarRAG Elite is a professional, market-ready RAG (Retrieval-Augmented Generation) application designed for students and researchers. It allows users to upload PDF notes and interact with them using a state-of-the-art AI tutor that can explain complex topics and generate Mermaid.js mind maps.

## 🚀 Key Features
* **Interactive RGB UI**: A sleek dark-mode interface with Yellow-Red RGB interactive glow effects.
* **Pinned Header**: All controls (Language, Sync, Clear, Upload) are locked in a professional gray-white header line.
* **Multi-Format Tutoring**: Get explanations in English, Hindi, or Hinglish with automated mind-map generation.
* **Responsive Design**: Fully optimized for both Desktop and Mobile views with zero gaps.
* **Social Integration**: One-click access to developer profiles (LinkedIn, GitHub, Substack).

## 🛠️ Tech Stack & Models
* **Core Framework**: LangChain (for RAG orchestration).
* **AI Model**: `llama-3.3-70b-versatile` via **Groq API** (Ultra-fast response).
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace).
* **Vector Store**: FAISS (for efficient local semantic search).
* **Frontend**: Streamlit (with custom CSS for ChatGPT-like experience).

## 📋 Prerequisites
- Python 3.10+ 
- Groq API Key

## 📥 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/ankitmodanwall/ScholarRAG.git](https://github.com/ankitmodanwall/ScholarRAG.git)
   cd ScholarRAG
