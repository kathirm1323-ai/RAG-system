## 📚 RAG Document Question Answering System

A Retrieval-Augmented Generation system that allows users to upload PDF documents and ask questions based on the document content.

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM_API-orange?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented_Generation-purple?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-yellow?style=for-the-badge)

---

## 🔄 Workflow

```text
User Uploads PDF
        ↓
PDF Text Extraction
        ↓
Text Chunking
        ↓
SentenceTransformer
        ↓
Generate Embeddings
        ↓
Store / Search Document Chunks
        ↓
User Asks Question
        ↓
Convert Question into Embedding
        ↓
Semantic Search
        ↓
Retrieve Relevant Chunks
        ↓
Send Question + Context to Groq LLM
        ↓
Generate Final Answer
        ↓
Display Answer to User
