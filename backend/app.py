import os
import uuid
import math
import re
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (allows frontend on GitHub Pages to connect)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)


@app.route('/health')
def health_check():
    return "OK", 200


@app.route('/')
def index():
    return jsonify({"status": "RAG Backend is running", "endpoints": ["/upload_multi", "/ask", "/history", "/status"]})


# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm_client = None


def get_llm_client():
    global llm_client
    if llm_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Please set it in Environment Variables.")
        llm_client = Groq(api_key=GROQ_API_KEY)
    return llm_client


# Global variables
def tokenize(text):
    return re.findall(r'\w+', text.lower())


# =====================================================================
# Lightweight Search Database with source metadata (BM25)
# =====================================================================
class ChromaVectorDB:
    """Lightweight, pure-Python search database replacing ChromaDB with BM25 retrieval."""

    def __init__(self, collection_name=None):
        self.collection_name = collection_name or f"collection_{uuid.uuid4().hex[:8]}"
        self.documents = []  # list of dicts: {"text": str, "source_file": str, "tokens": list}
        self.idf = {}
        self.vocab = set()

    def add_chunks(self, chunks, source_file):
        """Add text chunks with source file metadata."""
        if not chunks:
            return
        for chunk in chunks:
            tokens = tokenize(chunk)
            self.documents.append({
                "text": chunk,
                "source_file": source_file,
                "tokens": tokens
            })
        self._update_idf()

    def _update_idf(self):
        """Recalculate IDF for all terms in the vocabulary."""
        N = len(self.documents)
        if N == 0:
            self.idf = {}
            self.vocab = set()
            return
        
        df = Counter()
        for doc in self.documents:
            unique_terms = set(doc["tokens"])
            for term in unique_terms:
                df[term] += 1
        
        self.vocab = set(df.keys())
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (N - freq + 0.5) / (freq + 0.5)) + 1

    def remove_source(self, source_file):
        """Remove every indexed chunk belonging to a document."""
        original_count = len(self.documents)
        self.documents = [
            doc for doc in self.documents
            if doc["source_file"] != source_file
        ]
        self._update_idf()
        return original_count - len(self.documents)

    def search(self, query, top_k=5):
        """Search across all chunks using BM25 relevance scoring."""
        if not self.documents:
            return []
        
        query_tokens = tokenize(query)
        if not query_tokens:
            return [{"text": doc["text"], "source_file": doc["source_file"]} for doc in self.documents[:top_k]]
        
        k1 = 1.5
        b = 0.75
        avg_doc_len = sum(len(doc["tokens"]) for doc in self.documents) / len(self.documents)
        
        scores = []
        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_len = len(doc["tokens"])
            doc_token_counts = Counter(doc["tokens"])
            
            for term in query_tokens:
                if term in self.vocab:
                    tf = doc_token_counts[term]
                    idf = self.idf.get(term, 0.0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / (avg_doc_len or 1.0)))
                    score += idf * (numerator / denominator)
            
            scores.append((score, idx))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top_results = []
        for score, idx in scores[:top_k]:
            doc = self.documents[idx]
            top_results.append({
                "text": doc["text"],
                "source_file": doc["source_file"]
            })
            
        return top_results

    def get_chunk_count(self):
        return len(self.documents)


# =====================================================================
# Global State (Unified Knowledge Base)
# =====================================================================
class GlobalKnowledgeBase:
    def __init__(self):
        self.db = ChromaVectorDB(collection_name="global_knowledge")
        self.history = []
        self.uploaded_files = [] # List of unique filenames

    def clear_history(self):
        self.history = []

    def remove_file(self, filename):
        removed_chunks = self.db.remove_source(filename)
        self.uploaded_files = [
            uploaded_file for uploaded_file in self.uploaded_files
            if uploaded_file != filename
        ]
        self.clear_history()
        return removed_chunks

global_kb = GlobalKnowledgeBase()


# =====================================================================
# File text extractors
# =====================================================================
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_text_from_docx(filepath):
    try:
        import docx
        doc = docx.Document(filepath)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except ImportError:
        raise ValueError("python-docx is required for DOCX support. Install it with: pip install python-docx")


def extract_text_from_file(filepath, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.txt':
        return extract_text_from_txt(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .txt, .docx")


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# =====================================================================
# Answer generation
# =====================================================================
def is_personal_conversation(query):
    """Recognize only direct greetings or identity questions, never word fragments."""
    normalized_query = " ".join(tokenize(query))
    direct_greetings = {
        "hello", "hi", "hey", "greetings", "helo", "heyo",
        "how are you", "what s up"
    }
    return (
        normalized_query in direct_greetings
        or normalized_query == "who am i"
        or normalized_query == "what is my name"
        or normalized_query.startswith("my name is ")
    )


def generate_local_fallback(context):
    """Return a useful extract when the external LLM service is unavailable."""
    if not context:
        return "Please upload some documents first."

    sources = []
    sentences = []
    for chunk in context:
        source = chunk.get("source_file", "the uploaded document")
        if source not in sources:
            sources.append(source)
        sentences.extend(re.split(r"(?<=[.!?])\s+", chunk.get("text", "")))

    summary = " ".join(sentence.strip() for sentence in sentences[:3] if sentence.strip())
    if not summary:
        summary = "I found the document, but could not extract readable text from the relevant section."

    return f"Based on {', '.join(sources)}: {summary}"


def generate_answer(query, context, history):
    # Build context text with source annotations
    has_context = False
    if context and isinstance(context[0], dict):
        has_context = True
        context_parts = []
        for chunk in context:
            source = chunk.get("source_file", "unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Source: {source}]\n{text}")
        context_text = "\n\n---\n\n".join(context_parts)
    else:
        context_text = ""

    # History buffer (last 10 messages)
    history_text = ""
    for msg in history[-10:]:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Handle direct greetings and identity questions separately from document requests.
    if is_personal_conversation(query):
        prompt = f"""You are the RAG Executive AI. 
1. The user may be introducing themselves or asking about their identity.
2. Review the Past Conversation carefully. **PRIORITIZE the most recent name** shared by the user in the latest messages.
3. If they just said "My name is X", that is their current name. Disregard any older or conflicting names from previous turns.
4. If their name is known, address them directly.
5. If not known, be professional and ask how you can help with their documents.

Past Conversation:
{history_text}

New Message: {query}
Answer:"""
    elif has_context:
        prompt = f"""You are a professional assistant analyzing documents.

INSTRUCTIONS:
1. Understand the user's plain-English request and answer it using the given Context. Each context chunk is labeled with [Source: filename].
2. If the user asks to explain, summarize, describe, or give an overview of a document, provide a clear high-level summary of its content.
3. If the answer only comes from one file, that's fine — do NOT force content from both files.
4. If the context does not contain the answer, say "I cannot find information about this in the uploaded documents."
5. Do not discuss the user's name or identity unless the question directly asks about name or identity.
6. Keep answers concise, professional, and easy to understand.

Context:
{context_text}

Question: {query}
Answer:"""
    else:
        prompt = f"""You are a professional assistant. 
1. If the context is empty, say "Please upload some documents first."
2. Keep answers concise and professional.
    
Context:
(No context available for this module yet)

Past Conversation:
{history_text}

Question: {query}
Answer:"""

    print(f"DEBUG: Generating answer for query: '{query}'")
    
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        answer = response.choices[0].message.content
    except Exception as error:
        print(f"WARNING: Groq generation failed; using local fallback: {error}")
        answer = generate_local_fallback(context)

    print(f"DEBUG: LLM Response: '{answer}'")
    return answer


# =====================================================================
# Routes
# =====================================================================

@app.route('/upload', methods=['POST'])
@app.route('/upload_multi', methods=['POST'])
def upload_multi():
    """Upload any number of files. All are added to the global knowledge base."""
    files = request.files.getlist('files')

    # Fallback if sent as 'file'
    if not files and 'file' in request.files:
        files = request.files.getlist('file')

    # Filter out empty filenames
    files = [f for f in files if f.filename != '']

    if not files:
        return jsonify({"error": "No valid files provided"}), 400

    new_filenames = []
    saved_paths = []
    total_chunks_added = 0

    try:
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_paths.append(filepath)

            try:
                text = extract_text_from_file(filepath, filename)
            except ValueError as e:
                # Clean up and return
                for p in saved_paths:
                    if os.path.exists(p): os.remove(p)
                return jsonify({"error": f"Error processing {filename}: {str(e)}"}), 400

            if not text.strip():
                for p in saved_paths:
                    if os.path.exists(p): os.remove(p)
                return jsonify({"error": f"Empty or unreadable file: {filename}"}), 400

            chunks = split_text_into_chunks(text)
            
            # Add to global KB
            global_kb.db.add_chunks(chunks, source_file=filename)
            total_chunks_added += len(chunks)
            new_filenames.append(filename)
            
            # Track unique filenames
            if filename not in global_kb.uploaded_files:
                global_kb.uploaded_files.append(filename)

        # Clean up disk
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)

        return jsonify({
            "message": f"Successfully indexed {total_chunks_added} chunks from {len(new_filenames)} files!",
            "filenames": new_filenames
        })

    except Exception as e:
        for p in saved_paths:
            if os.path.exists(p): os.remove(p)
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route('/ask', methods=['POST'])
def ask():
    try:
        req = request.get_json()
        query = req.get('question')

        if not query:
            return jsonify({"error": "No question provided"}), 400

        # Search global knowledge base
        relevant_chunks = global_kb.db.search(query, top_k=5)
        answer = generate_answer(query, relevant_chunks, global_kb.history)

        global_kb.history.append({"role": "user", "content": query})
        global_kb.history.append({"role": "assistant", "content": answer})

        # Include source info in response
        sources_used = list(set(chunk["source_file"] for chunk in relevant_chunks)) if relevant_chunks else []
        return jsonify({"answer": answer, "sources": sources_used})
    except Exception as e:
        import traceback
        print("ERROR in /ask:")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": global_kb.history})


@app.route('/delete_history_item', methods=['POST'])
def delete_history_item():
    req = request.get_json()
    pair_index = req.get('pair_index')

    start_idx = pair_index * 2
    if start_idx < len(global_kb.history) - 1:
        del global_kb.history[start_idx:start_idx + 2]
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Invalid index"}), 400


@app.route('/clear_history', methods=['POST'])
def clear_history_route():
    global_kb.clear_history()
    return jsonify({"status": "success"})


@app.route('/delete_file', methods=['POST'])
def delete_file():
    req = request.get_json() or {}
    filename = secure_filename(req.get('filename', ''))

    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    if filename not in global_kb.uploaded_files:
        return jsonify({"error": "File not found"}), 404

    removed_chunks = global_kb.remove_file(filename)
    return jsonify({
        "status": "success",
        "removed_file": filename,
        "removed_chunks": removed_chunks,
        "history_cleared": True
    })


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "uploaded_files": global_kb.uploaded_files,
        "history_count": len(global_kb.history) // 2,
        "total_chunks": global_kb.db.get_chunk_count()
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
