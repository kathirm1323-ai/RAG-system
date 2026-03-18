# Advanced RAG System - Executive Intelligence

A high-performance Retrieval-Augmented Generation (RAG) system with a premium, modern UI. This application allows you to index PDF documents and have intelligent conversations about their content.

## ✨ Features

- **Document Indexing**: Drag-and-drop PDF upload for instant vectorization and analysis.
- **Premium UI**: Sleek dark-mode interface with gold accents and glassmorphism aesthetics.
- **Scrollable History Sidebar**: Keep track of all your queries in a dedicated, scrollable sidebar.
- **Individual Deletion**: Manage your history by deleting specific Q&A pairs directly from the sidebar.
- **Smooth Navigation**: Clicking a history item smoothly scrolls the chat window to the original message and highlights it with a gold pulse.
- **Session Persistence**: The system automatically detects if a document is already indexed on page load, keeping the chat active even after a refresh.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (Replace in `app.py`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kathirm1323-ai/RAG-system.git
   cd RAG-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the UI:
   Navigate to `http://127.0.0.1:5000` in your browser.

## 🛠️ Technology Stack
- **Backend**: Flask (Python)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Groq (Llama 3.1)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), and JavaScript (ES6)
