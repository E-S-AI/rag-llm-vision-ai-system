
# libraries we needed to installed
!pip install -qU langchain langchain-community langchain-core llama-cpp-python chromadb sentence-transformers pypdf unstructured[all-docs]
!pip install -q transformers==4.44.2 einops torch torchvision pillow

import os
import torch
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM , AutoTokenizer

# Part 1 of the task : Rag pipline
print("\n[PART 1: RAG Pipeline")

# 1.1 Igent & Chunk
pdf_path="sample_document.pdf"
if not os.path.exists(pdf_path):
    os.system('wget -q -O sample_document.pdf "https://arxiv.org/pdf/1706.03762.pdf"')

print(f"Loading document: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
print(f"Document split into {len(chunks)} chunks.")

# 1.2 Embed & Index (ChromaDB)
print("Initializing Local Embeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")

# 1.3 Local LLM Setup (Mistral 7B)
model_path = "./mistral.gguf"
if not os.path.exists(model_path):
    print("Downloading Mistral-7B model...")
    os.system('wget -q -O mistral.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf')

print("Loading Mistral-7B LLM into Memory...")
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=2048,
    temperature=0.1,
    verbose=False
)

# 1.4 Retrieval & QA Generation
query = "Based on the document, what is the role of the attention mechanism?"
print(f"\nUser Query: {query}")

# Explicit Retrieval for absolute stability
retrieved_docs = vector_db.similarity_search(query, k=2)
context_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""You are a helpful and precise AI assistant.
Use ONLY the following context to answer the question. If you don't know, say "I don't know".

Context:
{context_text}

Question: {query}

Answer:"""

answer = llm.invoke(prompt)
source_page = retrieved_docs[0].metadata.get('page', 'Unknown')

print(f"System Answer: {answer.strip()}")
print(f"Citation: Extracted from Page {source_page}")


# Part 2 : Vision Pipline
print("\n" + "="*50)
print("\n[PART 2: Vision Pipeline")


# 2.1 Load Image
img_path = "sample_invoice.png"
if not os.path.exists(img_path):
    os.system('wget -q -O sample_invoice.png "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"')

image = Image.open(img_path)
print("Invoice Image Loaded.")

# 2.2 Local VLM Setup (Moondream2)
print("Loading Moondream2 Vision Model...")
vlm_model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

tokenizer = AutoTokenizer.from_pretrained(vlm_model_id, revision=revision)
vlm_model = AutoModelForCausalLM.from_pretrained(
    vlm_model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16
).to("cuda")

# 2.3 Visual Extraction
vision_prompt = "What is the total amount due on this invoice? Please read the numbers carefully."
print(f"\nVision Task: {vision_prompt}")

enc_image = vlm_model.encode_image(image)
vision_answer = vlm_model.answer_question(enc_image, vision_prompt, tokenizer)

print(f"Vision AI Extracted Value: {vision_answer.strip()}")