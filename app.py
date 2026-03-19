import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm_client = Groq(api_key=GROQ_API_KEY)
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chat_history = []

class SimpleVectorDB:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    def add_chunks(self, chunks):
        self.chunks = chunks
        self.embeddings = embedding_model.encode(chunks)
    def search(self, query, top_k=3):
        if len(self.embeddings) == 0:
            return []
        query_embedding = embedding_model.encode([query])[0]
        similarities = []
        for i, chunk_emb in enumerate(self.embeddings):
            sim = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            similarities.append((sim, self.chunks[i]))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for sim, chunk in similarities[:top_k]]

class DocumentContext:
    def __init__(self, filename):
        self.filename = filename
        self.db = SimpleVectorDB()
        self.history = []

# Global Knowledge Base: filename -> DocumentContext
knowledge_base = {}

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_answer(query, context, history):
    context_text = "\n\n---\n\n".join(context)
    # Include past chat history in prompt for better continuity if needed
    history_text = ""
    for msg in history[-4:]: # include last 2 QA pairs
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    # Standard greeting detection
    greetings = ["hello", "hi", "hey", "greetings", "helo", "heyo", "how are you"]
    if query.lower() in greetings and not context:
        return "Greetings! I am the RAG Executive Interface. I am ready to analyze your document data once context is provided. How can I assist you today?"

    prompt = f"""You are a professional assistant. 
1. Use ONLY the given Context below to answer the Question.
2. If the context is empty or does not contain the answer, say "I cannot find information about this in the current module."
3. Keep answers concise and professional.
    
Context:
{context_text if context_text else "(No context available for this module yet)"}

Past Conversation:
{history_text}

Question: {query}
Answer:"""

    print(f"DEBUG: Generating answer for query: '{query}'")
    print(f"DEBUG: Context length: {len(context_text)}")
    
    response = llm_client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    print(f"DEBUG: LLM Response: '{answer}'")
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        text = extract_text_from_pdf(filepath)
        if not text.strip():
             return jsonify({"error": "Empty or unreadable PDF"}), 400
             
        chunks = split_text_into_chunks(text)
        
        # Create or update document context
        if filename not in knowledge_base:
            knowledge_base[filename] = DocumentContext(filename)
        
        knowledge_base[filename].db.add_chunks(chunks)
        
        # Clean up memory/disk
        os.remove(filepath)
        
        return jsonify({
            "message": f"Successfully indexed {len(chunks)} chunks in {filename}!",
            "filename": filename
        })

@app.route('/ask', methods=['POST'])
def ask():
    req = request.get_json()
    query = req.get('question')
    filename = req.get('filename') # Specify which document to query

    if not query:
        return jsonify({"error": "No question provided"}), 400
    if not filename or filename not in knowledge_base:
        return jsonify({"error": "Please select a document first"}), 400
        
    ctx = knowledge_base[filename]
    relevant_chunks = ctx.db.search(query)
    answer = generate_answer(query, relevant_chunks, ctx.history)
    
    ctx.history.append({"role": "user", "content": query})
    ctx.history.append({"role": "assistant", "content": answer})
    
    return jsonify({"answer": answer})

@app.route('/history', methods=['GET'])
def get_history():
    filename = request.args.get('filename')
    if not filename or filename not in knowledge_base:
        return jsonify({"history": []})
    return jsonify({"history": knowledge_base[filename].history})

@app.route('/delete_history_item', methods=['POST'])
def delete_history_item():
    req = request.get_json()
    filename = req.get('filename')
    pair_index = req.get('pair_index')

    if not filename or filename not in knowledge_base:
        return jsonify({"error": "Invalid document"}), 400
        
    history = knowledge_base[filename].history
    start_idx = pair_index * 2
    if start_idx < len(history) - 1:
        del history[start_idx:start_idx + 2]
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Invalid index"}), 400

@app.route('/clear_history', methods=['POST'])
def clear_history_route():
    req = request.get_json()
    filename = req.get('filename')
    if filename and filename in knowledge_base:
        knowledge_base[filename].history = []
    return jsonify({"status": "success"})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "documents": [
            {"filename": name, "history_count": len(ctx.history) // 2} 
            for name, ctx in knowledge_base.items()
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
