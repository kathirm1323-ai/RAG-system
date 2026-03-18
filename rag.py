import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import List

# =====================================================================
# Configuration
# =====================================================================
# Get your Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# We use SentenceTransformer to convert text into numbers (embeddings)
# 'all-MiniLM-L6-v2' is a small, fast, and local embedding model
print("Loading embedding model (this may take a moment on the first run)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Groq LLM client
llm_client = Groq(api_key=GROQ_API_KEY)


# =====================================================================
# 1. Read and Extract PDF
# =====================================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a PDF file and extracts all text."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# =====================================================================
# 2. Split Text into Chunks
# =====================================================================
def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits text into smaller chunks so we don't exceed LLM context limits.
    Overlap helps ensure we don't cut important sentences in half.
    """
    words = text.split()
    chunks = []
    
    # Loop over the words and group them into chunks
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        
    return chunks


# =====================================================================
# 3. Simple Vector Database
# =====================================================================
class SimpleVectorDB:
    """A minimal database that stores text chunks and their embeddings in memory."""
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        
    def add_chunks(self, chunks: List[str]):
        """Converts chunks to embeddings and stores them."""
        self.chunks = chunks
        print(f"Generating embeddings for {len(chunks)} chunks...")
        # embedding_model.encode() returns a list of numerical vectors
        self.embeddings = embedding_model.encode(chunks)
        
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Finds the most relevant chunks for a given query using Cosine Similarity."""
        # Step 1: Convert the query into an embedding
        query_embedding = embedding_model.encode([query])[0]
        
        similarities = []
        # Step 2: Compare query embedding against all saved chunk embeddings
        for i, chunk_emb in enumerate(self.embeddings):
            # Calculate Cosine Similarity using numpy
            # formula: dot_product(A, B) / (norm(A) * norm(B))
            sim = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            similarities.append((sim, self.chunks[i]))
            
        # Step 3: Sort by highest similarity score
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Step 4: Return the best 'top_k' chunks
        return [chunk for sim, chunk in similarities[:top_k]]


# =====================================================================
# 4. Generate Answer using LLM
# =====================================================================
def generate_answer(query: str, retrieved_context: List[str]) -> str:
    """Sends the context and question to Groq LLM to generate an answer."""
    # Combine chunks into a single readable text block
    context_text = "\n\n---\n\n".join(retrieved_context)
    
    # Prompt template enforcing the LLM to strictly use the provided context
    prompt = f"""You are a helpful and honest assistant. 
Use ONLY the given Context below to answer the Question. 
If the answer is not present in the context, say "I don't know based on the provided document". Do not make up answers.

Context:
{context_text}

Question: {query}
Answer:"""

    print("Thinking and generating answer with Groq LLM...")
    
    # We use Llama-3 available on Groq as it is very fast
    response = llm_client.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 # Low temperature so it doesn't get overly creative
    )
    
    return response.choices[0].message.content


# =====================================================================
# Main Execution Flow
# =====================================================================
def main():
    # --- SETUP ---
    # Please put your real PDF filename here!
    pdf_file = "sample.pdf" 
    
    if not os.path.exists(pdf_file):
        print(f"Error: '{pdf_file}' not found.")
        print("Please place a PDF file named 'sample.pdf' in the same folder as this script.")
        return

    # 1. Read PDF
    print(f"\nReading {pdf_file}...")
    pdf_text = extract_text_from_pdf(pdf_file)
    
    if not pdf_text.strip():
        print("Error: Could not extract text from the PDF. It might be scanned or empty.")
        return
        
    # 2. Split into chunks
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(pdf_text, chunk_size=300, overlap=50)

    # 3. Create Vector DB and store embeddings
    db = SimpleVectorDB()
    db.add_chunks(chunks)
    
    print("\nIndexing complete! You can now ask questions.")
    
    # Interactive Q&A Loop
    while True:
        question = input("\nAsk a question about the PDF (or type 'quit' to exit): ").strip()
        if question.lower() == 'quit':
            break
        if not question:
            continue
            
        # 4. Find relevant chunks
        print("\nFinding relevant information...")
        relevant_chunks = db.search(question, top_k=3)
        
        # 5. Generate Answer
        answer = generate_answer(question, relevant_chunks)
        
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print("="*50)
        print(answer)
        print("="*50)

if __name__ == "__main__":
    main()
