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
    def reset(self):
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

db = SimpleVectorDB()

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

def generate_answer(query, context):
    context_text = "\n\n---\n\n".join(context)
    # Include past chat history in prompt for better continuity if needed
    history_text = ""
    for msg in chat_history[-4:]: # include last 2 QA pairs
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    prompt = f"""You are a helpful assistant. Use ONLY the given Context below to answer the Question.
    
Context:
{context_text}

Past Conversation:
{history_text}

Question: {query}
Answer:"""

    response = llm_client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global chat_history
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
        db.reset()
        db.add_chunks(chunks)
        
        # Clean up memory/disk
        os.remove(filepath)
        
        chat_history = [] # Reset history on new document
        
        return jsonify({"message": f"Successfully indexed {len(chunks)} chunks!"})

@app.route('/ask', methods=['POST'])
def ask():
    req = request.get_json()
    query = req.get('question')
    if not query:
        return jsonify({"error": "No question provided"}), 400
    if not db.chunks:
        return jsonify({"error": "Please upload a document first"}), 400
        
    relevant_chunks = db.search(query)
    answer = generate_answer(query, relevant_chunks)
    
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    
    return jsonify({"answer": answer})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": chat_history})

@app.route('/delete_history_item/<int:pair_index>', methods=['POST'])
def delete_history_item(pair_index):
    global chat_history
    # Each pair is 2 items: [User Message, Assistant Message]
    # pair_index 0 is the oldest pair (indices 0, 1)
    # pair_index 1 is the next pair (indices 2, 3)
    start_idx = pair_index * 2
    if start_idx < len(chat_history) - 1:
        del chat_history[start_idx:start_idx + 2]
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Invalid index"}), 400

@app.route('/clear_history', methods=['POST'])
def clear_history_route():
    global chat_history
    chat_history = []
    return jsonify({"status": "success"})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "ready": bool(db.chunks),
        "history_count": len(chat_history) // 2
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
