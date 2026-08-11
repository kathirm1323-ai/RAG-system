# 📚 RAG Document Question Answering System

## 🚀 Introduction

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions based on the document content. The system retrieves the most relevant information from the PDF and uses an LLM to generate accurate, context-based answers.

🌐 **Live Demo:** [Try the RAG System](YOUR_LIVE_LINK_HERE)

---

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM_API-orange?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented_Generation-purple?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-yellow?style=for-the-badge)


---

## 🔄 RAG Workflow

📤 **PDF Upload**
→ 📄 Extract document text
→ ✂️ Split text into chunks
→ 🧠 Generate vector embeddings
→ 💾 Store document embeddings
→ ❓ User asks a question
→ 🔢 Convert question into embedding
→ 🔍 Perform semantic search
→ 📚 Retrieve relevant document chunks
→ 🤖 Send context + question to Groq LLM
→ ✅ Generate the final answer

---

## ✨ Key Features

* 📚 Ask questions directly from PDF documents
* 🧠 Understand document meaning using embeddings
* 🔍 Retrieve the most relevant content
* 🤖 Generate answers using Groq LLM
* 🎯 Reduce irrelevant and hallucinated responses
* ⚡ Fast backend processing with FastAPI
