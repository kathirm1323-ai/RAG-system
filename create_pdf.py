from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=15)
pdf.cell(200, 10, txt="Welcome to the RAG System PDF!", ln=1, align='C')
pdf.set_font("Arial", size=12)
text = """
Retrieval-Augmented Generation (RAG) is a technique that bridges the gap between large language models and specific, private, or dynamically changing data.
A RAG system typically consists of two main components: a retriever and a generator.
The retriever searches a database or a document collection for relevant information based on a user's query.
It uses vector embeddings, which are numerical representations of text, to measure semantic similarity between the query and the documents.
Once the relevant documents are found, the generator - usually a large language model - uses these documents as additional context to produce an accurate, detailed, and up-to-date answer.
This system significantly reduces hallucinations, ensures the answers are grounded in the provided factual data, and removes the need to constantly retrain models on new information.
"""
pdf.multi_cell(0, 10, text)
pdf.output("sample.pdf")
print("Successfully created sample.pdf")
