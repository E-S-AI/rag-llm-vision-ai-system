# Local AI Document Understanding System (RAG + Vision)

# Overview

This project implements a fully local AI system for:

- Retrieval-Augmented Generation (RAG)
- Vision-based document understanding

The system processes:
- PDF documents (native)
- Scanned documents (via vision model)
- Images (invoices)

It extracts information, builds a searchable knowledge base, and answers user queries with citations — all without using any external APIs.

---

# Architecture
PDF / Image
↓
Text Extraction / Vision Encoding
↓
Chunking
↓
Embeddings
↓
Vector Database (Chroma)
↓
Retriever
↓
Local LLM (Mistral)
↓
Answer + Citation


---

# Technologies Used

- LangChain (pipeline orchestration)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Mistral 7B GGUF (local LLM via llama.cpp)
- Moondream2 (vision-language model)
- PyTorch (model execution)

---

# Features

# RAG Pipeline
- Load and process PDF documents
- Split text into semantic chunks
- Generate embeddings locally
- Store vectors in ChromaDB
- Retrieve relevant context
- Generate answers using a local LLM
- Provide **source citations (page number)

---

# Vision Pipeline
- Load invoice image
- Use a local vision-language model (Moondream2)
- Extract information directly from the image
- Answer visual questions (e.g., total amount)

---

# How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt


2. Run the script
python Task_Titan_Tech.py

------

Fully Local Setup
No OpenAI / external APIs used
All models run locally
Suitable for privacy-sensitive applications

Project Structure
project/
│
├── ai_system.ipynb
├── chroma_db/
├── mistral.gguf
├── sample_document.pdf
├── sample_invoice.png
├── requirements.txt
└── README.md

